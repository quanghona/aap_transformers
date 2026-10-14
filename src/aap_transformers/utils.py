import base64
import io
import os
import tempfile
from typing import Any, Dict

import cv2
import httpx
import numpy as np
import soundfile as sf
import soxr
from aap_core.types import MediaReference
from PIL import Image

# transformers' BaseVideoProcessor class default; used when the loaded video
# processor carries no explicit num_frames.
DEFAULT_NUM_FRAMES = 8


def media_ref_to_image_part(ref: MediaReference) -> Dict[str, Any]:
    """Convert a resolved MediaReference to a transformers chat-template image part.

    base64 refs are decoded to a PIL image here, at prepare time, so a corrupt
    payload fails with a clear error before the model call instead of inside the
    processor's batch image loading; transformers' load_image accepts PIL images
    directly. URL refs pass through as {"type": "image", "url": ...} and are
    fetched by the processor when the conversation is tokenized (same contract
    as the llama-index integration).

    Other modalities have their own part types and their own mappers (audio:
    media_ref_to_audio_part); this helper refuses non-image refs rather than
    silently mislabeling them.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.

    Returns:
        Dict[str, Any]: A native content part for a message's content list.

    Raises:
        ValueError: If the ref is not an image, or its payload cannot be decoded.
    """
    if ref["content_type"] != "image":
        raise ValueError(
            f"media_ref_to_image_part got content_type={ref['content_type']!r}; "
            "only image refs are mapped today."
        )
    if ref["kind"] == "url":
        return {"type": "image", "url": ref["value"]}
    try:
        image = Image.open(io.BytesIO(base64.b64decode(ref["value"])))
        image.load()
    except Exception as exc:  # noqa: BLE001 - normalized into a ValueError contract
        raise ValueError(
            f"base64 image payload could not be decoded as an image "
            f"(mime_type={ref['mime_type']!r}): {exc}"
        ) from exc
    return {"type": "image", "image": image}


def media_ref_to_audio_part(
    ref: MediaReference, sampling_rate: int = 16_000
) -> Dict[str, Any]:
    """Convert a resolved MediaReference to a transformers chat-template audio part.

    base64 refs are decoded here to a mono float32 numpy array resampled to
    ``sampling_rate``, at prepare time: transformers' own string loaders
    (torchcodec/librosa) are optional extras that may not be installed, while
    an ndarray passes through its ``load_audio`` untouched - no decode, no
    resampling. So both must happen here. This mirrors
    ``transformers.audio_utils.load_audio_as``, which pairs the same
    soundfile+soxr stack.

    URL refs pass through as {"type": "audio", "url": ...} and are fetched by
    the processor when the conversation is tokenized, which requires
    transformers' audio extra (librosa or torchcodec). Decoding URLs here
    instead would duplicate that loader and narrow the codec set to soundfile's;
    the same contract as the image mapper.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.
        sampling_rate (int): Target rate, the model's feature extractor's
            sampling_rate so array inputs match what string inputs would be
            resampled to. Models' extractors default to 16 kHz.

    Returns:
        Dict[str, Any]: A native content part for a message's content list.

    Raises:
        ValueError: If the ref is not audio, or its payload cannot be decoded.
    """
    if ref["content_type"] != "audio":
        raise ValueError(
            f"media_ref_to_audio_part got content_type={ref['content_type']!r}; "
            "only audio refs are mapped by this helper."
        )
    if ref["kind"] == "url":
        return {"type": "audio", "url": ref["value"]}
    try:
        with io.BytesIO(base64.b64decode(ref["value"])) as buf, sf.SoundFile(buf) as f:
            audio = f.read(dtype="float32")
            original_rate = f.samplerate
    except Exception as exc:  # noqa: BLE001 - normalized into a ValueError contract
        raise ValueError(
            f"base64 audio payload could not be decoded as audio "
            f"(mime_type={ref['mime_type']!r}): {exc}"
        ) from exc
    if audio.ndim > 1:  # multichannel -> mono, what every audio encoder expects
        audio = audio.mean(axis=1)
    if original_rate != sampling_rate:
        audio = soxr.resample(audio, original_rate, sampling_rate, quality="HQ")
    return {"type": "audio", "audio": audio}


# Mime subtype -> filename suffix. Only cosmetic: the cv2/FFmpeg demuxer probes
# the content, not the name. Unknown subtypes keep the .mp4 default.
_VIDEO_SUFFIXES = {
    "quicktime": ".mov",
    "webm": ".webm",
    "avi": ".avi",
    "mpeg": ".mpg",
    "mpg": ".mpg",
    "x-flv": ".flv",
    "x-ms-wmv": ".wmv",
    "3gpp": ".3gp",
    "x-matroska": ".mkv",
}


def _sample_frames(frames: list, num_frames: int) -> "np.ndarray":
    """Uniformly reduce decoded BGR frames to an RGB (num_frames, H, W, 3) array."""
    total = len(frames)
    indices = np.unique(np.linspace(0, total - 1, min(num_frames, total), dtype=int))
    stacked = np.stack([frames[i] for i in indices])
    return np.ascontiguousarray(stacked[:, :, :, ::-1])  # BGR -> RGB


def _decode_video(path: str, num_frames: int) -> "np.ndarray":
    """Decode one video file into a uniformly sampled RGB frame array with cv2.

    Sequential read + subsample rather than CAP_PROP_POS_FRAMES seeks: seeking
    only lands on keyframes, and a single sequential pass is exact regardless
    of GOP structure. Videos longer than the frame budget are capped by
    reading at most ``stride * num_frames`` frames, so cost stays bounded by
    the sample target, not the clip length.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(
                f"video payload could not be opened by OpenCV ({path!r}); "
                "its codec may not be built into this opencv-python wheel."
            )
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(total // num_frames, 1) if total > num_frames else 1
        limit = num_frames * stride
        frames = []
        for _ in range(limit):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            raise ValueError(
                f"video at {path!r} decoded to zero frames; the container "
                "opened but no frame could be read."
            )
        return _sample_frames(frames, num_frames)
    finally:
        capture.release()


def media_ref_to_video_part(
    ref: MediaReference, num_frames: int = DEFAULT_NUM_FRAMES
) -> Dict[str, Any]:
    """Convert a resolved MediaReference to a transformers chat-template video part.

    Unlike the image and audio mappers, URL refs cannot pass through as strings
    here. transformers' own string-video loaders need torchcodec (the default
    backend) or PyAV (torchvision's fallback), neither of which is a guaranteed
    install, while its ndarray path bypasses decoding entirely. So every payload
    is decoded at this layer with cv2 (opencv-python ships FFmpeg in its wheels)
    and delivered as {"type": "video", "video": ndarray} of shape
    (frames, H, W, 3) uint8 RGB - the same contract as the audio ndarray.

    Frames are sampled uniformly down to ``num_frames`` here because the
    processor never gets the chance: array inputs skip its own sampling (it
    only samples strings it decodes itself), and loading every frame of a real
    clip would blow memory before the model saw one. This mirrors what a
    string input would have been sampled to via the processor's num_frames.

    base64 refs (including local paths, which core resolves to base64) are
    written to a temporary file because cv2 cannot capture from an in-memory
    buffer. URL refs are fetched with httpx first, for the same reason
    transformers' load_video does.

    Args:
        ref (MediaReference): The resolved reference from BaseLLMChain._media_parts.
        num_frames (int): Frame budget; the loaded video processor's num_frames
            so array inputs match what string inputs would be sampled to.

    Returns:
        Dict[str, Any]: A native content part for a message's content list.

    Raises:
        ValueError: If the ref is not video, or its payload cannot be decoded.
    """
    if ref["content_type"] != "video":
        raise ValueError(
            f"media_ref_to_video_part got content_type={ref['content_type']!r}; "
            "only video refs are mapped by this helper."
        )
    if ref["kind"] == "url":
        try:
            response = httpx.get(ref["value"], follow_redirects=True, timeout=60.0)
            response.raise_for_status()
            data = response.content
        except httpx.HTTPError as exc:
            raise ValueError(
                f"video URL could not be fetched ({ref['value']!r}): {exc}"
            ) from exc
    else:
        try:
            data = base64.b64decode(ref["value"])
        except Exception as exc:  # noqa: BLE001 - normalized into ValueError
            raise ValueError(
                f"base64 video payload could not be decoded "
                f"(mime_type={ref['mime_type']!r}): {exc}"
            ) from exc
    subtype = ref["mime_type"].split("/")[-1]
    suffix = _VIDEO_SUFFIXES.get(subtype, ".mp4")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"media{suffix}")
        with open(path, "wb") as handle:
            handle.write(data)
        try:
            video = _decode_video(path, num_frames)
        except ValueError as exc:
            raise ValueError(
                f"video payload could not be decoded as video "
                f"(mime_type={ref['mime_type']!r}, kind={ref['kind']!r}): {exc}"
            ) from exc
    return {"type": "video", "video": video}
