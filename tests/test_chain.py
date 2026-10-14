"""Offline structural tests for aap_transformers.chain media input support.

No network, model download, or GPU: tests exercise _prepare_conversation
assembly, capability detection from loaded-artifact metadata, and the media
gate only. Chains are built with fake (formatter, model) tuples, so no
from_pretrained call ever happens. The Hugging Face lookup is stubbed by an
autouse fixture so detection is deterministic offline.
"""

import base64
import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
from aap_core.types import AgentMessage, ContentType, MediaReference
from PIL import Image

import aap_transformers.chain as chain_module
import aap_transformers.utils as utils_module
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_transformers.utils import (
    DEFAULT_NUM_FRAMES,
    media_ref_to_audio_part,
    media_ref_to_image_part,
    media_ref_to_video_part,
)


def png_bytes(color: str = "red", size: Tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(sample_rate: int = 16_000, seconds: float = 0.05) -> bytes:
    buf = io.BytesIO()
    n = int(sample_rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    sf.write(
        buf, np.sin(2 * np.pi * 440 * t).astype("float32"), sample_rate, format="WAV"
    )
    return buf.getvalue()


PNG_DATA = png_bytes()
PNG_B64 = base64.b64encode(PNG_DATA).decode("ascii")
WAV_DATA = wav_bytes()
WAV_B64 = base64.b64encode(WAV_DATA).decode("ascii")


def mp4_bytes(frames: int = 12, size=(32, 32)) -> bytes:
    """Generate a tiny lossy mp4 whose frame i is uniformly gray i*16.

    Written with cv2 so the fixture needs no binary blobs in the repo; mp4v is
    lossy, so pixel assertions must allow a tolerance, and index coverage is
    asserted on _sample_frames-style synthetic data instead.
    """
    import cv2

    path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size
    )
    for i in range(frames):
        writer.write(np.full((*size[::-1], 3), i * 16 % 256, dtype=np.uint8))
    writer.release()
    with open(path, "rb") as handle:
        data = handle.read()
    os.remove(path)
    return data


class FakeResponse:
    """Stand-in for an httpx.Response returned by a stubbed get."""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "stubbed failure", request=None, response=self  # type: ignore[arg-type]
            )


@pytest.fixture
def mp4(tmp_path_factory) -> bytes:
    """Session-cheap generated mp4 fixture, written once per use."""
    return mp4_bytes()


class FakeInputs(dict):
    """Mimics the BatchFeature returned by apply_chat_template(return_dict=True).

    Subclasses dict because _generate_response unpacks it into model.generate(**...).
    """

    def __init__(self, input_ids: torch.Tensor):
        super().__init__(input_ids=input_ids)

    @property
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]

    def to(self, device):
        return self


class FakeTokenizer:
    """Stand-in for AutoTokenizer; records the conversations it is asked to format."""

    def __init__(self, name_or_path: str = ""):
        self.name_or_path = name_or_path
        self.calls: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        return FakeInputs(torch.zeros(1, 5, dtype=torch.long))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["answer"]


class FakeFeatureExtractor:
    """Stand-in for a feature extractor; its presence marks an audio pipeline."""

    def __init__(self, sampling_rate: int = 16_000):
        self.sampling_rate = sampling_rate


class FakeVideoProcessor:
    """Stand-in for a video processor; its presence marks a video pipeline."""

    def __init__(self, num_frames: int = DEFAULT_NUM_FRAMES):
        self.num_frames = num_frames


class FakeProcessor:
    """Stand-in for AutoProcessor; carries each pipeline only when capable.

    image_processor marks vision, feature_extractor marks audio, and
    video_processor marks video - exactly the attributes
    _resolve_capabilities probes on a real processor.
    """

    def __init__(
        self,
        tokenizer: Optional[FakeTokenizer] = None,
        vision: bool = True,
        audio: bool = False,
        video: bool = False,
        sampling_rate: int = 16_000,
        num_frames: int = DEFAULT_NUM_FRAMES,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        if vision:
            self.image_processor = object()
        if audio:
            self.feature_extractor = FakeFeatureExtractor(sampling_rate)
        if video:
            self.video_processor = FakeVideoProcessor(num_frames)
        self.calls: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        return FakeInputs(torch.zeros(1, 5, dtype=torch.long))


class FakeModel:
    def __init__(self):
        self.generated: List[Dict[str, Any]] = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generated.append(kwargs)
        return torch.zeros(1, 8, dtype=torch.long)


class HuggingFaceStub:
    """Stand-in for the model-card lookup: canned results plus call log."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[ContentType, bool]] = {}
        self.calls: List[str] = []

    def __call__(self, repo_id: str) -> Dict[ContentType, bool]:
        self.calls.append(repo_id)
        return self.results.get(repo_id, {})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep the default lookup inert so no test can reach the Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


@pytest.fixture
def hf(monkeypatch):
    """Install a recording Hugging Face stub over no_network's default."""
    stub = HuggingFaceStub()
    monkeypatch.setattr(chain_module, "huggingface_capabilities", stub)
    return stub


def make_chain(
    formatter: Any = None,
    model: Any = None,
    **kwargs,
) -> ChatCausalMultiTurnsChain:
    if formatter is None:
        formatter = FakeProcessor(tokenizer=FakeTokenizer())
    if model is None:
        model = FakeModel()
    return ChatCausalMultiTurnsChain(
        model=(formatter, model),
        device="cpu",
        system_prompt="You are helpful.",
        user_prompt_template="{query}",
        **kwargs,
    )


def user_message_of(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    return conversation[1]  # [0] is the system message


def image_refs(content: Any) -> List[Dict[str, Any]]:
    assert isinstance(content, list), f"expected content parts, got {content!r}"
    return [part for part in content if part.get("type") == "image"]


def audio_refs(content: Any) -> List[Dict[str, Any]]:
    assert isinstance(content, list), f"expected content parts, got {content!r}"
    return [part for part in content if part.get("type") == "audio"]


def video_refs(content: Any) -> List[Dict[str, Any]]:
    assert isinstance(content, list), f"expected content parts, got {content!r}"
    return [part for part in content if part.get("type") == "video"]


class TestMediaRefToImagePart:
    def test_url_ref_becomes_url_part(self):
        ref: MediaReference = {
            "content_type": "image",
            "kind": "url",
            "value": "https://example.com/cat.png",
            "mime_type": "image/png",
        }
        assert media_ref_to_image_part(ref) == {
            "type": "image",
            "url": "https://example.com/cat.png",
        }

    def test_base64_ref_decodes_to_pil_image(self):
        ref: MediaReference = {
            "content_type": "image",
            "kind": "base64",
            "value": PNG_B64,
            "mime_type": "image/png",
        }
        part = media_ref_to_image_part(ref)
        assert part["type"] == "image"
        assert isinstance(part["image"], Image.Image)
        assert (
            part["image"].tobytes()
            == Image.open(io.BytesIO(PNG_DATA)).convert("RGB").tobytes()
        )

    def test_corrupt_base64_raises_value_error(self):
        ref: MediaReference = {
            "content_type": "image",
            "kind": "base64",
            "value": base64.b64encode(b"not an image").decode("ascii"),
            "mime_type": "image/png",
        }
        with pytest.raises(ValueError, match="could not be decoded"):
            media_ref_to_image_part(ref)

    def test_non_image_ref_is_refused(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "url",
            "value": "https://example.com/sound.wav",
            "mime_type": "audio/wav",
        }
        with pytest.raises(ValueError, match="only image refs"):
            media_ref_to_image_part(ref)


class TestMediaRefToAudioPart:
    def test_url_ref_becomes_url_part(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "url",
            "value": "https://example.com/sound.wav",
            "mime_type": "audio/wav",
        }
        assert media_ref_to_audio_part(ref) == {
            "type": "audio",
            "url": "https://example.com/sound.wav",
        }

    def test_base64_ref_decodes_to_float32_array(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "base64",
            "value": WAV_B64,
            "mime_type": "audio/wav",
        }
        part = media_ref_to_audio_part(ref)
        assert part["type"] == "audio"
        assert part["audio"].dtype == np.float32
        assert part["audio"].ndim == 1

    def test_base64_ref_is_resampled_to_target_rate(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "base64",
            "value": base64.b64encode(
                wav_bytes(sample_rate=48_000, seconds=1.0)
            ).decode("ascii"),
            "mime_type": "audio/wav",
        }
        part = media_ref_to_audio_part(ref, sampling_rate=16_000)
        assert len(part["audio"]) == 16_000

    def test_stereo_is_downmixed_to_mono(self):
        buf = io.BytesIO()
        sf.write(buf, np.zeros((16_000, 2), dtype="float32"), 16_000, format="WAV")
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "base64",
            "value": base64.b64encode(buf.getvalue()).decode("ascii"),
            "mime_type": "audio/wav",
        }
        assert media_ref_to_audio_part(ref)["audio"].ndim == 1

    def test_corrupt_base64_raises_value_error(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "base64",
            "value": base64.b64encode(b"not audio").decode("ascii"),
            "mime_type": "audio/wav",
        }
        with pytest.raises(ValueError, match="could not be decoded"):
            media_ref_to_audio_part(ref)

    def test_non_audio_ref_is_refused(self):
        ref: MediaReference = {
            "content_type": "image",
            "kind": "url",
            "value": "https://example.com/cat.png",
            "mime_type": "image/png",
        }
        with pytest.raises(ValueError, match="only audio refs"):
            media_ref_to_audio_part(ref)


class TestMediaRefToVideoPart:
    """Video is decoded at our layer: transformers' string loaders need
    torchcodec/PyAV, which are not guaranteed installs, so unlike image and
    audio there is no URL passthrough to test -- URLs are fetched here."""

    def test_base64_ref_decodes_to_frame_array(self, mp4):
        ref: MediaReference = {
            "content_type": "video",
            "kind": "base64",
            "value": base64.b64encode(mp4).decode("ascii"),
            "mime_type": "video/mp4",
        }
        part = media_ref_to_video_part(ref, num_frames=4)
        assert part["type"] == "video"
        video = part["video"]
        assert video.ndim == 4 and video.shape[-1] == 3
        assert video.dtype == np.uint8
        assert video.shape[0] <= 4

    def test_url_ref_is_fetched_and_decoded(self, mp4, monkeypatch):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return FakeResponse(mp4)

        monkeypatch.setattr(utils_module.httpx, "get", fake_get)
        ref: MediaReference = {
            "content_type": "video",
            "kind": "url",
            "value": "https://example.com/clip.mp4",
            "mime_type": "video/mp4",
        }
        part = media_ref_to_video_part(ref, num_frames=4)
        assert calls == ["https://example.com/clip.mp4"]
        assert part["video"].ndim == 4

    def test_failed_url_fetch_raises_value_error(self, monkeypatch):
        def fake_get(url, **kwargs):
            return FakeResponse(b"", status_code=404)

        monkeypatch.setattr(utils_module.httpx, "get", fake_get)
        ref: MediaReference = {
            "content_type": "video",
            "kind": "url",
            "value": "https://example.com/gone.mp4",
            "mime_type": "video/mp4",
        }
        with pytest.raises(ValueError, match="could not be fetched"):
            media_ref_to_video_part(ref)

    def test_frames_sampled_to_budget(self, mp4):
        ref: MediaReference = {
            "content_type": "video",
            "kind": "base64",
            "value": base64.b64encode(mp4).decode("ascii"),
            "mime_type": "video/mp4",
        }
        few = media_ref_to_video_part(ref, num_frames=2)["video"]
        many = media_ref_to_video_part(ref, num_frames=8)["video"]
        assert few.shape[0] == 2 < many.shape[0]

    def test_uniform_sampling_covers_first_and_last_frame(self):
        # Index coverage asserted without a codec: synthetic marker frames,
        # because mp4v is lossy and pixel values cannot be asserted exactly.
        from aap_transformers.utils import _sample_frames

        frames = [np.full((4, 4, 3), i, np.uint8) for i in range(30)]
        out = _sample_frames(frames, 8)
        markers = out[:, 0, 0, 0]
        assert markers[0] == 0 and markers[-1] == 29
        assert len(set(markers.tolist())) == 8

    def test_bgr_converted_to_rgb(self, mp4):
        # A pure-blue BGR source must arrive as RGB [..., 255] modulo codec loss.
        import cv2

        path = tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16))
        for _ in range(4):
            writer.write(
                np.tile(np.array([[[255, 0, 0]]], np.uint8), (16, 16, 1))
            )
        writer.release()
        with open(path, "rb") as handle:
            data = handle.read()
        os.remove(path)
        ref: MediaReference = {
            "content_type": "video",
            "kind": "base64",
            "value": base64.b64encode(data).decode("ascii"),
            "mime_type": "video/mp4",
        }
        video = media_ref_to_video_part(ref, num_frames=2)["video"]
        assert video[0, 0, 0, 2] > 200  # red channel last => RGB order
        assert video[0, 0, 0, 0] < 60  # blue channel first

    def test_corrupt_base64_raises_value_error(self):
        ref: MediaReference = {
            "content_type": "video",
            "kind": "base64",
            "value": base64.b64encode(b"not a video").decode("ascii"),
            "mime_type": "video/mp4",
        }
        with pytest.raises(ValueError, match="could not be decoded"):
            media_ref_to_video_part(ref)

    def test_non_video_ref_is_refused(self):
        ref: MediaReference = {
            "content_type": "audio",
            "kind": "url",
            "value": "https://example.com/sound.wav",
            "mime_type": "audio/wav",
        }
        with pytest.raises(ValueError, match="only video refs"):
            media_ref_to_video_part(ref)


class TestPrepareConversationTextOnly:
    """Regression: text-only behavior must be identical to before media support."""

    def test_plain_text_message(self):
        chain = make_chain()
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0] == {"role": "system", "content": "You are helpful."}
        assert user_message_of(conversation)["role"] == "user"
        assert user_message_of(conversation)["content"] == "hello"

    def test_template_with_context(self):
        chain = ChatCausalMultiTurnsChain(
            model=(FakeProcessor(), FakeModel()),
            device="cpu",
            system_prompt="s",
            user_prompt_template="{query} about {context_topic}",
        )
        msg = AgentMessage(query="q", context={"topic": "t"})
        assert (
            user_message_of(chain._prepare_conversation(msg))["content"] == "q about t"
        )

    def test_history_appended_after_user_message(self):
        chain = make_chain()
        chain.include_history = 2
        msg = AgentMessage(
            query="now", responses=[("user", "before"), ("agent", "answer")]
        )
        conversation = chain._prepare_conversation(msg)
        assert len(conversation) == 4
        assert conversation[2] == {"role": "user", "content": "before"}
        assert conversation[3] == {"role": "assistant", "content": "answer"}

    def test_tool_and_system_roles_mapped(self):
        chain = make_chain()
        chain.include_history = -1
        msg = AgentMessage(
            query="q", responses=[("tool", "t-out"), ("system", "s-out")]
        )
        conversation = chain._prepare_conversation(msg)
        assert conversation[2]["role"] == "tool"
        assert conversation[3]["role"] == "system"


class TestPrepareConversationWithImages:
    """Image media must become native content parts on the current user message."""

    def test_url_image_becomes_content_parts(self):
        chain = make_chain()
        msg = AgentMessage(
            query="what is this?",
            query_media=[("image", "https://example.com/cat.png")],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1] == {"type": "image", "url": "https://example.com/cat.png"}

    def test_local_path_image_becomes_pil_part(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(PNG_DATA)
        chain = make_chain()
        msg = AgentMessage(query="describe", query_media=[("image", str(f))])
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image"
        assert isinstance(content[1]["image"], Image.Image)

    def test_multiple_images_preserve_order(self):
        chain = make_chain()
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("image", "https://example.com/b.jpg"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert [part["url"] for part in image_refs(content)] == [
            "https://example.com/a.png",
            "https://example.com/b.jpg",
        ]

    def test_media_never_leaks_into_prompt_text(self):
        chain = make_chain()
        msg = AgentMessage(
            query="look",
            query_media=[("image", "https://example.com/cat.png")],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        text_parts = [part["text"] for part in content if part["type"] == "text"]
        assert text_parts == ["look"], "media repr leaked into the prompt template"

    def test_non_image_media_not_attached(self):
        chain = make_chain()
        msg = AgentMessage(
            query="hi", query_media=[("audio", "https://example.com/sound.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "hi"

    def test_missing_local_path_raises(self, tmp_path):
        chain = make_chain()
        msg = AgentMessage(
            query="oops", query_media=[("image", str(tmp_path / "absent.png"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)


class TestPrepareConversationWithAudio:
    """Audio media must become native content parts on the current user message."""

    def audio_chain(self, **kwargs) -> ChatCausalMultiTurnsChain:
        return make_chain(formatter=FakeProcessor(vision=False, audio=True), **kwargs)

    def test_url_audio_becomes_content_parts(self):
        chain = self.audio_chain()
        msg = AgentMessage(
            query="what do you hear?",
            query_media=[("audio", "https://example.com/sound.wav")],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[0] == {"type": "text", "text": "what do you hear?"}
        assert content[1] == {
            "type": "audio",
            "url": "https://example.com/sound.wav",
        }

    def test_base64_audio_becomes_array_part(self):
        chain = self.audio_chain()
        msg = AgentMessage(
            query="listen", query_media=[("audio", f"data:audio/wav;base64,{WAV_B64}")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[1]["type"] == "audio"
        assert isinstance(content[1]["audio"], np.ndarray)

    def test_local_path_audio_becomes_array_part(self, tmp_path):
        f = tmp_path / "sound.wav"
        f.write_bytes(WAV_DATA)
        chain = self.audio_chain()
        msg = AgentMessage(query="hear this?", query_media=[("audio", str(f))])
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[1]["type"] == "audio"
        assert isinstance(content[1]["audio"], np.ndarray)

    def test_extractor_sampling_rate_is_used(self):
        chain = make_chain(
            formatter=FakeProcessor(vision=False, audio=True, sampling_rate=24_000)
        )
        msg = AgentMessage(
            query="q",
            query_media=[("audio", f"data:audio/wav;base64,{WAV_B64}")],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(content[1]["audio"]) == int(0.05 * 24_000)

    def test_multiple_audios_preserve_order(self):
        chain = self.audio_chain()
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("audio", "https://example.com/a.wav"),
                ("audio", "https://example.com/b.wav"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert [part["url"] for part in audio_refs(content)] == [
            "https://example.com/a.wav",
            "https://example.com/b.wav",
        ]

    def test_images_precede_audios_in_content(self):
        chain = make_chain(formatter=FakeProcessor(vision=True, audio=True))
        msg = AgentMessage(
            query="both?",
            query_media=[
                ("audio", "https://example.com/sound.wav"),
                ("image", "https://example.com/cat.png"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert [part["type"] for part in content] == ["text", "image", "audio"]

    def test_media_never_leaks_into_prompt_text(self):
        chain = self.audio_chain()
        msg = AgentMessage(
            query="listen",
            query_media=[("audio", "https://example.com/sound.wav")],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        text_parts = [part["text"] for part in content if part["type"] == "text"]
        assert text_parts == ["listen"], "media repr leaked into the prompt template"


class TestPrepareConversationWithVideo:
    """Video media must become frame-array content parts on the current user message."""

    def video_chain(self, **kwargs) -> ChatCausalMultiTurnsChain:
        return make_chain(formatter=FakeProcessor(vision=False, video=True), **kwargs)

    def test_base64_video_becomes_array_part(self, mp4):
        chain = self.video_chain()
        msg = AgentMessage(
            query="watch",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[0] == {"type": "text", "text": "watch"}
        assert content[1]["type"] == "video"
        assert isinstance(content[1]["video"], np.ndarray)
        assert content[1]["video"].ndim == 4

    def test_local_path_video_becomes_array_part(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(mp4_bytes())
        chain = self.video_chain()
        msg = AgentMessage(query="see this?", query_media=[("video", str(f))])
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[1]["type"] == "video"
        assert isinstance(content[1]["video"], np.ndarray)

    def test_url_video_is_fetched_not_passed_through(self, mp4, monkeypatch):
        """The one deliberate break from the image/audio URL passthrough:
        transformers cannot decode string videos without torchcodec/PyAV, so
        the fetch happens at our layer."""
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return FakeResponse(mp4)

        monkeypatch.setattr(utils_module.httpx, "get", fake_get)
        chain = self.video_chain()
        msg = AgentMessage(
            query="clip?", query_media=[("video", "https://example.com/clip.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert calls == ["https://example.com/clip.mp4"]
        assert isinstance(content[1]["video"], np.ndarray)

    def test_processor_num_frames_is_used(self, mp4):
        chain = make_chain(
            formatter=FakeProcessor(vision=False, video=True, num_frames=3)
        )
        msg = AgentMessage(
            query="q",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content[1]["video"].shape[0] == 3

    def test_multiple_videos_preserve_order(self, mp4):
        chain = self.video_chain()
        b64 = base64.b64encode(mp4).decode()
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("video", f"data:video/mp4;base64,{b64}"),
                ("video", f"data:video/mp4;base64,{b64}"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert [part["type"] for part in content] == ["text", "video", "video"]

    def test_media_order_is_text_image_audio_video(self, mp4):
        chain = make_chain(
            formatter=FakeProcessor(vision=True, audio=True, video=True)
        )
        msg = AgentMessage(
            query="all?",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}"),
                ("audio", "https://example.com/sound.wav"),
                ("image", "https://example.com/cat.png"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert [part["type"] for part in content] == [
            "text",
            "image",
            "audio",
            "video",
        ]

    def test_media_never_leaks_into_prompt_text(self, mp4):
        chain = self.video_chain()
        msg = AgentMessage(
            query="watch",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        text_parts = [part["text"] for part in content if part["type"] == "text"]
        assert text_parts == ["watch"], "media repr leaked into the prompt template"


class TestGenerateResponse:
    def test_tokenizer_path_unchanged_for_text_only_processor(self):
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(tokenizer=tokenizer, vision=False)
        chain = make_chain(formatter=processor)
        conversation = [{"role": "user", "content": "hi"}]
        conversation, output, has_tool, usage = chain._generate_response(conversation)
        assert output == "answer"
        assert has_tool is False
        assert conversation[-1] == {"role": "assistant", "content": "answer"}
        assert usage == {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        }
        # Text-only formatter is the tokenizer, as before media support.
        assert tokenizer.calls and not processor.calls

    def test_processor_path_receives_content_parts(self):
        processor = FakeProcessor()
        chain = make_chain(formatter=processor)
        msg = AgentMessage(
            query="what?", query_media=[("image", "https://example.com/cat.png")]
        )
        conversation = chain._prepare_conversation(msg)
        chain._generate_response(conversation)
        assert len(processor.calls) == 1
        sent_conversation, sent_kwargs = processor.calls[0]
        assert sent_kwargs["tokenize"] is True
        assert sent_kwargs["return_dict"] is True
        assert image_refs(sent_conversation[1]["content"])

    def test_audio_only_processor_routes_to_processor_not_tokenizer(self):
        """The feature pipeline needs the processor exactly like the pixel one."""
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(tokenizer=tokenizer, vision=False, audio=True)
        chain = make_chain(formatter=processor)
        conversation = chain._prepare_conversation(
            AgentMessage(
                query="q", query_media=[("audio", "https://example.com/a.wav")]
            )
        )
        chain._generate_response(conversation)
        assert len(processor.calls) == 1
        assert tokenizer.calls == []
        assert audio_refs(processor.calls[0][0][1]["content"])

    def test_video_only_processor_routes_to_processor_not_tokenizer(self):
        """The video pipeline needs the processor exactly like the pixel one."""
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(tokenizer=tokenizer, vision=False, video=True)
        chain = make_chain(formatter=processor)
        conversation = chain._prepare_conversation(
            AgentMessage(
                query="q",
                query_media=[
                    (
                        "video",
                        f"data:video/mp4;base64,{base64.b64encode(mp4_bytes()).decode()}",
                    )
                ],
            )
        )
        chain._generate_response(conversation)
        assert len(processor.calls) == 1
        assert tokenizer.calls == []
        assert video_refs(processor.calls[0][0][1]["content"])

    def test_tool_call_detection(self):
        class ToolCallingModel(FakeModel):
            def generate(self, **kwargs):
                self.generated.append(kwargs)
                out = torch.zeros(1, 8, dtype=torch.long)
                return out

        chain = make_chain(model=ToolCallingModel())
        chain._tokenizer.batch_decode = lambda token_ids, skip_special_tokens=True: [
            "<tool_call>{}"
        ]
        conversation, output, has_tool, usage = chain._generate_response(
            [{"role": "user", "content": "hi"}]
        )
        assert has_tool is True


class TestCapabilityDetection:
    def test_vision_processor_is_decisive_true(self):
        chain = make_chain(formatter=FakeProcessor())
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": False,
            "video": False,
        }

    def test_processor_without_image_processor_is_decisive_false(self):
        chain = make_chain(formatter=FakeProcessor(vision=False))
        assert chain.detect_capabilities() == {
            "image": False,
            "audio": False,
            "video": False,
        }

    def test_audio_processor_is_decisive_per_modality(self):
        chain = make_chain(formatter=FakeProcessor(vision=False, audio=True))
        assert chain.detect_capabilities() == {
            "image": False,
            "audio": True,
            "video": False,
        }

    def test_video_processor_is_decisive_per_modality(self):
        chain = make_chain(formatter=FakeProcessor(vision=False, video=True))
        assert chain.detect_capabilities() == {
            "image": False,
            "audio": False,
            "video": True,
        }

    def test_omni_processor_reports_all_true(self):
        chain = make_chain(
            formatter=FakeProcessor(vision=True, audio=True, video=True)
        )
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": True,
            "video": True,
        }

    def test_plain_tokenizer_without_repo_id_is_undecidable(self):
        chain = make_chain(formatter=FakeTokenizer(name_or_path="local-weights"))
        assert chain.detect_capabilities() == {}

    def test_hf_card_negative_when_no_processor(self, hf):
        hf.results["Qwen/Qwen2.5-7B-Instruct"] = {"image": False}
        chain = make_chain(
            formatter=FakeTokenizer(name_or_path="Qwen/Qwen2.5-7B-Instruct")
        )
        assert chain.detect_capabilities() == {"image": False}

    def test_hf_card_decides_audio_without_processor(self, hf):
        """No processor loaded: the model card's task tags fill the audio gap."""
        hf.results["org/audio-repo"] = {"image": False, "audio": True}
        chain = make_chain(formatter=FakeTokenizer(name_or_path="org/audio-repo"))
        msg = AgentMessage(
            query="q", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(audio_refs(content)) == 1

    def test_provider_name_never_hits_hf(self, hf):
        chain = make_chain(formatter=FakeTokenizer(name_or_path="gemma3:12b"))
        assert chain.detect_capabilities() == {}
        assert hf.calls == []

    def test_local_directory_never_hits_hf(self, hf, tmp_path, monkeypatch):
        """A relative dir like "org/repo" is repo-id shaped but must not be looked up."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "org" / "repo").mkdir(parents=True)
        chain = make_chain(formatter=FakeTokenizer(name_or_path="org/repo"))
        assert chain.detect_capabilities() == {}
        assert hf.calls == []

    def test_processor_evidence_skips_hf_lookup(self, hf):
        hf.results["org/repo"] = {"image": False}
        chain = make_chain(
            formatter=FakeProcessor(tokenizer=FakeTokenizer(name_or_path="org/repo"))
        )
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": False,
            "video": False,
        }
        assert hf.calls == []

    def test_capabilities_are_cached(self, hf):
        hf.results["org/repo"] = {"image": False}
        chain = make_chain(formatter=FakeTokenizer(name_or_path="org/repo"))
        chain.detect_capabilities()
        chain.detect_capabilities()
        assert len(hf.calls) == 1

    def test_model_reassignment_invalidates_cache(self, hf):
        hf.results["org/repo"] = {"image": False}
        chain = make_chain(formatter=FakeTokenizer(name_or_path="org/repo"))
        chain.detect_capabilities()
        chain.model = (FakeTokenizer(name_or_path="org/repo"), FakeModel())
        chain.detect_capabilities()
        assert len(hf.calls) == 2

    def test_model_round_trip_preserves_processor(self):
        processor = FakeProcessor()
        chain = make_chain(formatter=processor)
        formatter, model = chain.model
        chain.model = (formatter, model)
        assert chain._processor is processor
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": False,
            "video": False,
        }


class TestMediaGate:
    def test_incapable_model_drops_media_with_warning(self, caplog):
        chain = make_chain(formatter=FakeProcessor(vision=False))
        msg = AgentMessage(
            query="pic?", query_media=[("image", "https://example.com/cat.png")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "pic?"
        assert "does not support image" in caplog.text

    def test_disabled_policy_overrides_vision_processor(self):
        chain = make_chain(
            formatter=FakeProcessor(),
            media_support={"image": "disabled"},
        )
        msg = AgentMessage(
            query="pic?", query_media=[("image", "https://example.com/cat.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "pic?"

    def test_enabled_policy_overrides_decisive_negative(self):
        chain = make_chain(
            formatter=FakeProcessor(vision=False),
            media_support={"image": "enabled"},
        )
        msg = AgentMessage(
            query="pic?", query_media=[("image", "https://example.com/cat.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(image_refs(content)) == 1

    def test_undecidable_assumes_capable_fail_loud(self):
        chain = make_chain(formatter=FakeTokenizer(name_or_path="local-weights"))
        msg = AgentMessage(
            query="pic?", query_media=[("image", "https://example.com/cat.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(image_refs(content)) == 1

    def test_incapable_model_drops_audio_with_warning(self, caplog):
        chain = make_chain(formatter=FakeProcessor())  # vision-only: audio False
        msg = AgentMessage(
            query="sound?", query_media=[("audio", "https://example.com/a.wav")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "sound?"
        assert "does not support audio" in caplog.text

    def test_disabled_policy_overrides_audio_processor(self):
        chain = make_chain(
            formatter=FakeProcessor(vision=False, audio=True),
            media_support={"audio": "disabled"},
        )
        msg = AgentMessage(
            query="sound?", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "sound?"

    def test_enabled_policy_overrides_decisive_audio_negative(self):
        chain = make_chain(
            formatter=FakeProcessor(),  # vision processor: audio decisively False
            media_support={"audio": "enabled"},
        )
        msg = AgentMessage(
            query="sound?", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(audio_refs(content)) == 1

    def test_incapable_model_drops_video_with_warning(self, caplog, mp4):
        chain = make_chain(formatter=FakeProcessor())  # vision-only: video False
        msg = AgentMessage(
            query="clip?",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "clip?"
        assert "does not support video" in caplog.text

    def test_disabled_policy_overrides_video_processor(self, mp4):
        chain = make_chain(
            formatter=FakeProcessor(vision=False, video=True),
            media_support={"video": "disabled"},
        )
        msg = AgentMessage(
            query="clip?",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert content == "clip?"

    def test_enabled_policy_overrides_decisive_video_negative(self, mp4):
        chain = make_chain(
            formatter=FakeProcessor(),  # vision processor: video decisively False
            media_support={"video": "enabled"},
        )
        msg = AgentMessage(
            query="clip?",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg))["content"]
        assert len(video_refs(content)) == 1


class TestInvokeEndToEnd:
    def test_invoke_with_image_delivers_parts_and_records_usage(self):
        processor = FakeProcessor()
        model = FakeModel()
        chain = make_chain(formatter=processor, model=model)
        msg = AgentMessage(
            query="what?", query_media=[("image", "https://example.com/cat.png")]
        )
        result = chain(msg)
        assert result.responses == [(chain.name, "answer")]
        assert result.origin == chain.name
        assert result.execution_result == "success"
        assert result.token_usage["total"]["input_tokens"] == 5
        assert result.token_usage["total"]["output_tokens"] == 3
        assert len(processor.calls) == 1
        assert image_refs(processor.calls[0][0][1]["content"])

    def test_invoke_text_only_model_drops_media_and_succeeds(self, caplog):
        processor = FakeProcessor(vision=False)
        chain = make_chain(formatter=processor)
        msg = AgentMessage(
            query="what?", query_media=[("image", "https://example.com/cat.png")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            result = chain(msg)
        assert result.responses == [(chain.name, "answer")]
        assert not image_refs_of_any_call(processor)

    def test_invoke_with_audio_delivers_parts_and_records_usage(self):
        processor = FakeProcessor(vision=False, audio=True)
        model = FakeModel()
        chain = make_chain(formatter=processor, model=model)
        msg = AgentMessage(
            query="what do you hear?",
            query_media=[("audio", "https://example.com/sound.wav")],
        )
        result = chain(msg)
        assert result.responses == [(chain.name, "answer")]
        assert result.execution_result == "success"
        assert len(processor.calls) == 1
        assert audio_refs(processor.calls[0][0][1]["content"])

    def test_invoke_audio_only_model_drops_image_keeps_audio(self, caplog):
        processor = FakeProcessor(vision=False, audio=True)
        chain = make_chain(formatter=processor)
        msg = AgentMessage(
            query="mixed?",
            query_media=[
                ("image", "https://example.com/cat.png"),
                ("audio", "https://example.com/sound.wav"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            result = chain(msg)
        assert result.execution_result == "success"
        content = processor.calls[0][0][1]["content"]
        assert not image_refs(content)
        assert len(audio_refs(content)) == 1

    def test_invoke_with_video_delivers_parts_and_records_usage(self, mp4):
        processor = FakeProcessor(vision=False, video=True)
        model = FakeModel()
        chain = make_chain(formatter=processor, model=model)
        msg = AgentMessage(
            query="what happens?",
            query_media=[
                ("video", f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}")
            ],
        )
        result = chain(msg)
        assert result.responses == [(chain.name, "answer")]
        assert result.execution_result == "success"
        assert len(processor.calls) == 1
        assert video_refs(processor.calls[0][0][1]["content"])

    def test_query_media_untouched_on_message(self):
        chain = make_chain()
        msg = AgentMessage(
            query="what?", query_media=[("image", "https://example.com/cat.png")]
        )
        chain._prepare_conversation(msg)
        assert msg.query_media == [("image", "https://example.com/cat.png")]


def image_refs_of_any_call(processor: FakeProcessor) -> List[Dict[str, Any]]:
    parts = []
    for conversation, _ in processor.calls:
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                parts += [p for p in content if p.get("type") == "image"]
    return parts
