import base64
import io
from typing import Any, Dict

import soxr
import soundfile as sf
from aap_core.types import MediaReference
from PIL import Image


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
