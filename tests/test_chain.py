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
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from aap_core.types import AgentMessage, ContentType, MediaReference
from PIL import Image

import aap_transformers.chain as chain_module
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_transformers.utils import media_ref_to_image_part


def png_bytes(color: str = "red", size: Tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


PNG_DATA = png_bytes()
PNG_B64 = base64.b64encode(PNG_DATA).decode("ascii")


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


class FakeProcessor:
    """Stand-in for AutoProcessor; carries an image_processor only when vision."""

    def __init__(self, tokenizer: Optional[FakeTokenizer] = None, vision: bool = True):
        self.tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        if vision:
            self.image_processor = object()
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
        assert chain.detect_capabilities() == {"image": True}

    def test_processor_without_image_processor_is_decisive_false(self):
        chain = make_chain(formatter=FakeProcessor(vision=False))
        assert chain.detect_capabilities() == {"image": False}

    def test_plain_tokenizer_without_repo_id_is_undecidable(self):
        chain = make_chain(formatter=FakeTokenizer(name_or_path="local-weights"))
        assert chain.detect_capabilities() == {}

    def test_hf_card_negative_when_no_processor(self, hf):
        hf.results["Qwen/Qwen2.5-7B-Instruct"] = {"image": False}
        chain = make_chain(
            formatter=FakeTokenizer(name_or_path="Qwen/Qwen2.5-7B-Instruct")
        )
        assert chain.detect_capabilities() == {"image": False}

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
        assert chain.detect_capabilities() == {"image": True}
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
        assert chain.detect_capabilities() == {"image": True}


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
