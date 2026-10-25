"""Contract tests for the TOON prettier (aap_core 0.3.1) through the Transformers adapter.

TOON is a **core-only** feature: ``DataFrameRetriever.from_pandas`` and
``DataFrameRetriever.from_dict`` accept ``prettier="toon"`` and encode through
``toon_format`` (declared by aap_core as ``toon-format>=1.0.0``). This adapter ships
no TOON code of its own, so what this module pins down is the boundary
``aap_transformers`` owns: a TOON payload must survive the trip

    DataFrameRetriever -> AgentMessage.context -> ChatCausalMultiTurnsChain._prepare_conversation
        -> tokenizer.apply_chat_template

byte-for-byte. The load-bearing risk is TOON's tabular header, which is
brace-bearing (``"[2]{sku,qty}:"``), meeting ``str.format`` interpolation at
``chain.py:191`` and then Jinja rendering inside ``apply_chat_template``. Values are
not re-scanned by either, so the payload must arrive intact; these tests fail if
that ever stops holding.

Serialization layers covered offline (verified, not assumed):
  1. ``str.format`` interpolation of ``format_kwargs()`` into the user template.
  2. Conversation assembly into ``TransformersChainMessage`` dicts.
  3. Real Transformers chat-template rendering: an in-memory
     ``PreTrainedTokenizerFast`` built from ``tokenizers`` with a Jinja template
     (no ``from_pretrained``, no weights, no Hub) renders the chain's own
     conversation, which is the same code path a checkpoint's template runs.
  4. ``_generate_response``'s formatter hand-off and ``invoke`` orchestration,
     with a fake model standing in for ``generate``.

Layer NOT covered: a real checkpoint's own chat template and processor path
(``AutoProcessor`` with image/feature/video pipelines). Both need a download, so
they are out of scope by design; the fake ``(formatter, model)`` tuple accepted by
``_bind_model`` reaches every branch that does not require weights.

Also documented here: ``aap_transformers.retriever`` is a 0-byte module, so this
adapter ships no ``RetrieverAdapter`` (see TestMissingRetrieverAdapterSurface).

Offline by design: no model download, no network, no HF Hub access.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
import toon_format
import torch
from aap_core.retriever import DataFrameRetriever
from aap_core.types import AgentMessage
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import aap_transformers.retriever as transformers_retriever_module
from aap_transformers.chain import ChatCausalMultiTurnsChain, TransformersChainMessage

# Minimum aap_core these contracts require: 0.3.1 routed prettier="toon" to
# toon_format, 0.3.2 made format_kwargs() join list-valued context into text.
TOON_MIN_AAP_CORE = (0, 3, 2)

# A brace-bearing table: the exact shape that would break if TOON ever landed in
# a template position instead of a value position.
SAMPLE_RECORDS = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]

# A minimal Jinja chat template, structurally like the ones checkpoints ship:
# role tag, then the raw content expression.
CHAT_TEMPLATE = (
    "{% for message in messages %}<<|{{ message['role'] }}|>>\n"
    "{{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<<|assistant|>>\n{% endif %}"
)


def resolved_aap_core_version() -> tuple:
    """Installed aap_core as a comparable tuple, tolerant of suffixes."""
    raw = version("aap_core")
    numbers: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers)


def toon_dataframe(records: List[Dict[str, Any]], data_key: str = "context.data"):
    """A DataFrameRetriever whose payload is TOON-encoded, via the public factory."""
    return DataFrameRetriever.from_pandas(
        pd.DataFrame(records), data_key=data_key, prettier="toon"
    )


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


class RecordingTokenizer:
    """Tokenizer stand-in: records every conversation it is asked to format."""

    name_or_path = ""

    def __init__(self, reply: str = "the answer"):
        self.reply = reply
        self.seen: List[List[TransformersChainMessage]] = []

    def apply_chat_template(self, conversation, **kwargs):
        self.seen.append(list(conversation))
        return FakeInputs(torch.zeros(1, 5, dtype=torch.long))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.reply]


class FakeModel:
    """Model stand-in: fixed-length output, no weights, no device transfer."""

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.zeros(1, 8, dtype=torch.long)


def make_real_tokenizer() -> PreTrainedTokenizerFast:
    """A genuine transformers tokenizer with a Jinja chat template, built in memory.

    Nothing is downloaded: the vocabulary is a single unknown token, which is
    enough because the property under test is template *rendering*, not token ids.
    """
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


class RealTemplateFormatter:
    """Delegates rendering to a real PreTrainedTokenizerFast and records the text.

    Deliberately carries no ``tokenizer``/``image_processor`` attribute, so
    ``_bind_model`` keeps it on the tokenizer-only branch instead of treating it
    as a multimodal processor.
    """

    name_or_path = "in-memory-jinja"

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerFast] = None):
        self.tokenizer_obj = (
            tokenizer if tokenizer is not None else make_real_tokenizer()
        )
        self.seen: List[List[TransformersChainMessage]] = []
        self.rendered: List[str] = []

    def apply_chat_template(self, conversation, **kwargs):
        self.seen.append(list(conversation))
        # Same conversation, string mode: what a checkpoint template would send
        # to the model. The tokenize=True result is returned untouched.
        self.rendered.append(
            self.tokenizer_obj.apply_chat_template(conversation, tokenize=False)
        )
        return FakeInputs(torch.zeros(1, 5, dtype=torch.long))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["[unused]"]


def make_chain(
    formatter: Any = None,
    model: Any = None,
    user_prompt_template: str = "{query}\n{context_data}",
    system_prompt: str = "You are helpful.",
    **kwargs,
) -> ChatCausalMultiTurnsChain:
    """Chain bound to a caller-supplied (formatter, model) pair, so no weights load."""
    return ChatCausalMultiTurnsChain(
        model=(
            formatter if formatter is not None else RecordingTokenizer(),
            model if model is not None else FakeModel(),
        ),
        device="cpu",
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        **kwargs,
    )


def user_content_of(
    chain: ChatCausalMultiTurnsChain, message: AgentMessage
) -> TransformersChainMessage:
    """The rendered user turn: conversation[0] is the system message."""
    entry = chain._prepare_conversation(message)[1]
    assert isinstance(entry["content"], str), (
        "expected a plain-text user turn; media parts would make content a list"
    )
    return entry


class TestResolvedCoreSupportsToon:
    """The upgrade claim: the aap_core this repo resolves actually enables TOON."""

    def test_toon_format_distribution_is_importable(self):
        """toon_format is importable and round-trips, so the core branch can run."""
        assert toon_format.encode({"a": 1}) == "a: 1"
        assert toon_format.decode("a: 1") == {"a": 1}
        assert toon_format.encode(["str1", "str2"]) == "[2]: str1,str2"

    def test_aap_core_version_meets_toon_requirement(self):
        """Resolved aap_core is >= 0.3.1, the release that enabled toon_format."""
        try:
            resolved = resolved_aap_core_version()
        except PackageNotFoundError as exc:  # pragma: no cover - env misconfig
            pytest.fail(f"aap_core is not installed in this environment: {exc}")
        assert resolved >= TOON_MIN_AAP_CORE, (
            f"aap_core {resolved} resolved from the environment does not enable "
            f"toon_format; aap_core>={'.'.join(map(str, TOON_MIN_AAP_CORE))} is required. "
            "Bump the dependency floor in pyproject.toml and re-run uv sync."
        )

    def test_from_pandas_routes_toon_to_toon_format(self):
        """prettier="toon" encodes records losslessly instead of falling to tabulate."""
        retriever = toon_dataframe(SAMPLE_RECORDS)
        message = retriever.retrieve(AgentMessage(query="q"))

        payload = message.context["data"]
        assert isinstance(payload, str), f"expected an encoded string, got {payload!r}"
        assert payload == "[2]{sku,qty}:\n  A-1,2\n  B-2,7"
        assert toon_format.decode(payload) == SAMPLE_RECORDS

    def test_from_dict_routes_toon_to_toon_format(self):
        """prettier="toon" on a dict encodes it losslessly."""
        data = {"warehouse": "north", "capacity": 128}
        retriever = DataFrameRetriever.from_dict(data, prettier="toon")
        message = retriever.retrieve(AgentMessage(query="q"))

        assert toon_format.decode(message.context["data"]) == data

    def test_toon_keyword_is_case_sensitive_and_degrades_silently(self):
        """Only the exact lowercase "toon" routes to toon_format.

        A misspelled prettier is passed to tabulate as a tablefmt, which does not
        reject unknown formats. This test documents the trap: a typo yields a
        plausible-looking string that is NOT TOON, with no exception raised.
        """
        typo = DataFrameRetriever.from_pandas(
            pd.DataFrame(SAMPLE_RECORDS), prettier="TOON"
        )
        payload = typo.retrieve(AgentMessage(query="q")).context["data"]

        assert "{" not in payload, (
            f"prettier='TOON' unexpectedly produced TOON output ({payload!r}); the "
            "routing condition in aap_core changed and this canary needs revisiting"
        )
        with pytest.raises(Exception):
            toon_format.decode(payload)


class TestToonPayloadIntoTransformersPrompt:
    """TOON context must reach the Transformers conversation verbatim."""

    def test_dataframe_toon_payload_reaches_user_content(self):
        message = toon_dataframe(SAMPLE_RECORDS).retrieve(
            AgentMessage(query="total qty?")
        )

        entry = user_content_of(make_chain(), message)

        assert entry["content"] == "total qty?\n[2]{sku,qty}:\n  A-1,2\n  B-2,7"

    def test_toon_tabular_header_braces_survive_str_format(self):
        """TOON's "{cols}" header must not be consumed as a format placeholder."""
        payload = toon_format.encode(SAMPLE_RECORDS)
        assert "{" in payload, (
            "TOON no longer emits brace-bearing headers; this regression canary "
            "no longer exercises the risk it was written for"
        )

        entry = user_content_of(
            make_chain(), AgentMessage(query="q", context={"data": payload})
        )

        assert payload in entry["content"], "brace-bearing header was mangled"
        assert entry["content"].count(payload) == 1, "payload must appear exactly once"

    def test_from_dict_toon_payload_reaches_user_content(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
        retriever = DataFrameRetriever.from_dict(data, prettier="toon")
        message = retriever.retrieve(AgentMessage(query="inventory"))

        entry = user_content_of(make_chain(), message)

        assert entry["content"] == "inventory\n" + toon_format.encode(data)
        assert toon_format.decode(entry["content"].split("\n", 1)[1]) == data

    def test_toon_payload_survives_message_flattening(self):
        """format_kwargs()/to_dict() keep the TOON string under context_data."""
        payload = toon_format.encode(SAMPLE_RECORDS)
        message = AgentMessage(query="q", context={"data": payload})

        assert message.format_kwargs()["context_data"] == payload
        assert message.to_dict()["context_data"] == payload

    def test_roles_are_system_then_user(self):
        payload = toon_format.encode(SAMPLE_RECORDS)
        conversation = make_chain()._prepare_conversation(
            AgentMessage(query="q", context={"data": payload})
        )

        assert [entry["role"] for entry in conversation] == ["system", "user"]

    def test_system_message_is_unaffected_by_toon_context(self):
        payload = toon_format.encode(SAMPLE_RECORDS)
        message = AgentMessage(query="q", context={"data": payload})
        conversation = make_chain(
            system_prompt="Rules: answer from {data} only"
        )._prepare_conversation(message)

        # The system prompt is not interpolated, so its literal braces stay put.
        assert conversation[0]["content"] == "Rules: answer from {data} only"
        assert payload in conversation[1]["content"]

    @pytest.mark.parametrize(
        "records,expected",
        [
            (
                [{"sku": "A-1", "qty": 2}],
                "[1]{sku,qty}:\n  A-1,2",
            ),
            (
                [
                    {"sku": "A-1", "qty": 2},
                    {"sku": "B-2", "qty": 7},
                    {"sku": "C-3", "qty": 0},
                ],
                "[3]{sku,qty}:\n  A-1,2\n  B-2,7\n  C-3,0",
            ),
            ([{"name": "café", "note": "你好"}], "[1]{name,note}:\n  café,你好"),
            (
                [{"id": 1, "meta": {"x": 1}}, {"id": 2, "meta": {"x": 2}}],
                "[2]{id,meta{x}}:\n  1,1\n  2,2",
            ),
            (
                [{"a": 1, "b": "line1\nline2"}],
                '[1]{a,b}:\n  1,"line1\\nline2"',
            ),
        ],
        ids=["single-row", "multi-row", "unicode", "nested-column", "multiline-value"],
    )
    def test_various_tables_are_delivered_verbatim(self, records, expected):
        payload = toon_format.encode(records)
        assert payload == expected, "TOON output drifted from the verified literal"

        entry = user_content_of(
            make_chain(), AgentMessage(query="q", context={"data": payload})
        )

        assert entry["content"] == "q\n" + payload
        assert toon_format.decode(entry["content"].split("\n", 1)[1]) == records

    def test_empty_dataframe_encodes_to_empty_table_token(self):
        """Edge case: zero rows must not blow up the prompt path."""
        retriever = DataFrameRetriever.from_pandas(
            pd.DataFrame(SAMPLE_RECORDS).iloc[0:0], prettier="toon"
        )
        message = retriever.retrieve(AgentMessage(query="q"))

        assert message.context["data"] == "[]"
        assert user_content_of(make_chain(), message)["content"] == "q\n[]"
        assert toon_format.decode("[]") == []

    def test_nan_becomes_null_and_stays_prompt_safe(self):
        """Known lossiness: NaN is encoded as TOON null, decoded as None."""
        records = [{"a": 1.0, "b": float("nan")}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="q"))
        payload = message.context["data"]

        assert payload == "[1]{a,b}:\n  1,null"
        assert toon_format.decode(payload) == [{"a": 1, "b": None}]
        assert payload in user_content_of(make_chain(), message)["content"]

    def test_toon_is_not_duplicated_through_replayed_history(self):
        """History turns are prior responses; the TOON table lands once, in turn 1.

        Note: include_history defaults to 0, yet _prepare_conversation still
        replays the whole history because ``responses[-0:]`` is
        ``responses[0:]`` (chain.py:218). That is a separate defect; what matters
        here is that it does not duplicate the TOON payload.
        """
        payload = toon_format.encode(SAMPLE_RECORDS)
        message = AgentMessage(
            query="q",
            context={"data": payload},
            responses=[("user", "earlier question"), ("ai", "earlier answer")],
        )

        conversation = make_chain()._prepare_conversation(message)

        assert conversation[1]["content"] == f"q\n{payload}"
        assert [entry["role"] for entry in conversation] == [
            "system",
            "user",
            "user",
            "assistant",
        ]
        assert all(
            payload not in str(entry["content"]) for entry in conversation[2:]
        ), "TOON leaked into replayed history"
        assert sum(str(e["content"]).count(payload) for e in conversation) == 1

    def test_toon_survives_real_jinja_chat_template_rendering(self):
        """The deepest layer reachable without weights: transformers' own renderer.

        ``apply_chat_template`` is what turns the conversation into the prompt a
        checkpoint sees. Rendering it with an in-memory tokenizer proves the
        brace-bearing table is not re-parsed or escaped by Jinja.
        """
        payload = toon_format.encode(SAMPLE_RECORDS)
        conversation = make_chain()._prepare_conversation(
            AgentMessage(query="q", context={"data": payload})
        )

        rendered = make_real_tokenizer().apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        assert payload in rendered, f"Jinja rendering mangled the table: {rendered!r}"
        assert rendered.count(payload) == 1
        assert rendered.endswith("<<|assistant|>>\n"), "generation prompt was appended"


class TestMissingRetrieverAdapterSurface:
    """``aap_transformers.retriever`` is a 0-byte module: no adapter ships here.

    Documentary evidence that this absence is intentional, not accidental:
    ``README.md`` lists ``retriever.py  RetrieverAdapter (planned)`` and
    ``aap_doc/docs/integrations/transformers/overview.md`` calls it "currently a
    placeholder; no implementation yet". The sibling page
    ``components/retriever.md`` documents a full ``RetrieverAdapter`` anyway
    (mkdocs API directive included), which is stale boilerplate - its own usage
    examples are TODO stubs. These tests therefore document the gap rather than
    fail on it: the contract the repo actually commits to is core-only retrieval.
    The list-context contract below is the forward-looking constraint a real
    adapter must honour.
    """

    def test_documented_adapter_name_is_not_importable(self):
        with pytest.raises(ImportError, match="RetrieverAdapter"):
            from aap_transformers.retriever import RetrieverAdapter  # noqa: F401

    def test_retriever_module_exposes_no_public_names(self):
        public = [
            name
            for name in dir(transformers_retriever_module)
            if not name.startswith("_")
        ]

        assert public == [], (
            f"aap_transformers.retriever unexpectedly exposes {public}; the module "
            "is documented as a placeholder, so update README/overview with it"
        )

    def test_list_valued_context_renders_as_text_not_repr(self):
        """Constraint on whoever implements the adapter: list context is text.

        ``AgentMessage.format_kwargs`` joins a list of strings into lines (core
        0.3.2), so a ``{context_key}`` placeholder interpolates the TOON table
        itself rather than a Python list repr. That is what lets a future
        transformers adapter store TOON documents as a list without mangling
        them - the shape the langchain/llamaindex adapters already produce.
        """
        payload = toon_format.encode(SAMPLE_RECORDS)
        message = AgentMessage(query="q", context={"data": [payload]})

        entry = user_content_of(make_chain(), message)

        assert entry["content"] == "q\n" + payload
        assert toon_format.decode(entry["content"].split("\n", 1)[1]) == SAMPLE_RECORDS


class TestToonEndToEnd:
    """Retriever -> AgentMessage -> chain.invoke, with no weights loaded."""

    def test_retrieve_then_invoke_delivers_toon_to_the_formatter(self):
        formatter = RecordingTokenizer(reply="9 units")
        chain = make_chain(formatter=formatter)
        message = toon_dataframe(SAMPLE_RECORDS)(AgentMessage(query="total qty?"))

        result = chain.invoke(message)

        assert (
            formatter.seen[0][1]["content"]
            == f"total qty?\n{toon_format.encode(SAMPLE_RECORDS)}"
        )
        assert result.responses == [("chain", "9 units")]
        assert result.execution_result == "success"
        assert result.context["data"] == toon_format.encode(SAMPLE_RECORDS), (
            "context must survive the chain"
        )

    def test_invoke_populates_token_usage_from_the_generated_batch(self):
        chain = make_chain()
        message = AgentMessage(
            query="q", context={"data": toon_format.encode(SAMPLE_RECORDS)}
        )

        result = chain.invoke(message)

        assert len(result.token_usage["steps"]) == 1
        assert result.token_usage["total"]["input_tokens"] == 5
        assert result.token_usage["total"]["output_tokens"] == 3
        assert result.token_usage["total"]["total_tokens"] == 8

    def test_invoke_through_a_real_chat_template_keeps_toon_verbatim(self):
        formatter = RealTemplateFormatter()
        chain = make_chain(formatter=formatter)
        payload = toon_format.encode(SAMPLE_RECORDS)

        result = chain.invoke(AgentMessage(query="q", context={"data": payload}))

        assert payload in formatter.rendered[0], (
            f"real Jinja rendering lost the table: {formatter.rendered[0]!r}"
        )
        assert formatter.seen[0][1]["content"] == f"q\n{payload}"
        assert result.execution_result == "success"
        assert result.context["data"] == payload

    def test_invoke_round_trips_toon_context_back_into_agent_message(self):
        """Round-trip: TOON in context -> prompt -> out again -> decodes."""
        payload = toon_format.encode(SAMPLE_RECORDS)
        chain = make_chain()
        message = chain.invoke(AgentMessage(query="q", context={"data": payload}))

        assert toon_format.decode(message.context["data"]) == SAMPLE_RECORDS
        assert toon_format.decode(message.to_dict()["context_data"]) == SAMPLE_RECORDS

    def test_conversation_entry_types_match_the_typeddict_contract(self):
        """TransformersChainMessage stays role/content; content is a plain str."""
        payload = toon_format.encode(SAMPLE_RECORDS)
        conversation = make_chain()._prepare_conversation(
            AgentMessage(query="q", context={"data": payload})
        )

        assert set(conversation[1].keys()) == {"role", "content"}
        assert isinstance(conversation[1]["content"], str)
        assert not isinstance(conversation[1]["content"], list), (
            "a str content means no media parts were attached, so the TOON table "
            "is not nested inside a content list"
        )
