import json
import os
import re
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Literal, Tuple

from aap_core import utils
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, ContentType, TokenUsage
from aap_core.utils import extract_repo_id, huggingface_capabilities
from pydantic import Field, PrivateAttr
from typing_extensions import TypedDict

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from aap_transformers.utils import media_ref_to_image_part


class TransformersChainMessage(TypedDict):
    role: str
    content: str | List[Dict[str, Any]]


class ChatCausalMultiTurnsChain(
    BaseCausalMultiTurnsChain[TransformersChainMessage, str]
):
    _model: Any
    _tokenizer: Any = PrivateAttr()  # AutoTokenizer has type of Unknown!
    _processor: Any = PrivateAttr(default=None)  # set when a processor was loadable
    _model_id: str | None = PrivateAttr(default=None)
    _capabilities_cache: Dict[ContentType, bool] | None = PrivateAttr(default=None)
    device: Literal["cpu", "cuda"] = Field(
        ..., description="Device the model to run on"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")
    _tool_dict: Dict[str, Callable] = PrivateAttr({})

    def __init__(
        self,
        model: str | os.PathLike | Tuple[Any, Any],
        tools: Sequence[Callable] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._bind_model(model)
        self.tools = tools

    def _bind_model(self, model: str | os.PathLike | Tuple[Any, Any]) -> None:
        """Load a pretrained identifier or bind a caller-supplied (formatter, model) pair.

        For identifiers, AutoProcessor is tried first: when it loads and carries
        an image_processor, the vision model class and the processor are used,
        because only the processor expands image parts into pixel inputs. Every
        other case keeps the previous tokenizer-only path unchanged. A processor
        that loads without an image_processor is still retained as decisive
        negative evidence for capability detection.
        """
        if isinstance(model, (str, os.PathLike)):
            identifier = os.fspath(model)
            self._model_id = (
                identifier if isinstance(identifier, str) else str(identifier)
            )
            try:
                processor = AutoProcessor.from_pretrained(self._model_id)
            except Exception:  # noqa: BLE001 - text-only models ship no processor config
                processor = None
            self._processor = processor
            if processor is not None and hasattr(processor, "image_processor"):
                self._tokenizer = processor.tokenizer
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, device_map=self.device
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                # drop device_map if running on CPU
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id, device_map=self.device
                )
            self._model.eval()
        else:
            formatter, self._model = model
            self._model_id = None
            if hasattr(formatter, "image_processor") or hasattr(formatter, "tokenizer"):
                self._processor = formatter
                self._tokenizer = getattr(formatter, "tokenizer", formatter)
            else:
                self._processor = None
                self._tokenizer = formatter
        self._capabilities_cache = None

    def detect_capabilities(self) -> Dict[ContentType, bool]:
        """Resolve per-modality input support from the loaded artifacts, not names.

        Sources, in priority order:
        1. What the user declared in media_support ("enabled"/"disabled"). The
           core gate applies that before this method runs, so an explicit
           declaration always wins.
        2. The processor actually loaded for this exact revision: presence of an
           image_processor is decisive in both directions - these are the very
           weights and pixel pipeline that will run.
        3. Hugging Face model-card metadata, for models identified by repo id
           when no processor was loadable (huggingface-hub is a hard dependency
           of transformers, so no extra is needed).

        Model-family names carry no weight here: variants of one family differ
        per modality, so name matching misreports exactly the cases that
        matter. Modalities no source reports on stay absent, and auto mode
        treats them as capable (fail loud).

        The result is cached and invalidated whenever the model is replaced,
        since source 3 costs a network request.
        """
        if self._capabilities_cache is None:
            self._capabilities_cache = self._resolve_capabilities()
        return dict(self._capabilities_cache)

    def _resolve_capabilities(self) -> Dict[ContentType, bool]:
        """Query the metadata sources once, filling gaps from the next source."""
        if self._processor is not None:
            return {"image": hasattr(self._processor, "image_processor")}
        repo_id = extract_repo_id(self._identifier_candidates())
        if repo_id is not None:
            return huggingface_capabilities(repo_id)
        return {}

    def _identifier_candidates(self) -> List[str]:
        """Model identifiers to test for a Hugging Face repo id, best first.

        Local paths are dropped here rather than left to the repo-id pattern:
        a relative directory like "models/qwen" is shaped exactly like a repo
        id and would otherwise cost a 404'd Hub request per lookup.
        """
        candidates: List[str] = []
        for value in (
            self._model_id,
            getattr(self._tokenizer, "name_or_path", None),
            getattr(getattr(self._model, "config", None), "_name_or_path", None),
        ):
            if isinstance(value, str) and not os.path.exists(value):
                candidates.append(value)
        return candidates

    @classmethod
    def extract_json(cls, input_str: str) -> List[str]:
        pattern = r"<tool_call>(?s:.*?)<\/tool_call>"
        matches = re.findall(pattern, input_str)
        return [
            match.strip()
            .replace("<tool_call>", "")
            .replace("</tool_call>", "")
            .replace("\\n", "")
            for match in matches
        ]

    def _prepare_conversation(
        self, message: AgentMessage
    ) -> List[TransformersChainMessage]:
        user_prompt = self.user_prompt_template.format(**message.format_kwargs())
        images = self._image_parts(message)
        user_content: str | List[Dict[str, Any]] = user_prompt
        if images:
            user_content = [
                {"type": "text", "text": user_prompt},
                *[media_ref_to_image_part(ref) for ref in images],
            ]
        conversation: List[TransformersChainMessage] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append({"role": "user", "content": response[1]})
            elif response[0] == "tool":
                conversation.append({"role": "tool", "content": response[1]})
            elif response[0] == "system":  # this shouldn't happen but just in case
                conversation.append({"role": "system", "content": response[1]})
            else:
                conversation.append({"role": "assistant", "content": response[1]})
        return conversation

    def _generate_response(
        self,
        conversation: List[TransformersChainMessage],
        **kwargs,
    ) -> Tuple[List[TransformersChainMessage], str, bool, TokenUsage]:
        # The processor owns the pixel pipeline: image parts in the content
        # lists are extracted and loaded by it when tokenize=True. It defaults
        # to tokenize=False (string rendering), unlike the tokenizer, so the
        # flag must be explicit. Without a vision processor the tokenizer path
        # is kept byte-for-byte identical to the pre-media behavior.
        formatter = (
            self._processor
            if self._processor is not None
            and hasattr(self._processor, "image_processor")
            else self._tokenizer
        )
        inputs = formatter.apply_chat_template(
            conversation,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = self._model.generate(**inputs.to(self.device), **kwargs)
        input_tokens = inputs.input_ids.shape[-1]
        output_tokens = output[:, inputs.input_ids.shape[-1] :].shape[-1]
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        output = self._tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )[0]
        output = utils.remove_thinking(output)
        # TODO: aware with multimodal output
        conversation.append({"role": "assistant", "content": output})
        return conversation, output, "<tool_call>" in output, usage

    def _process_tools(
        self,
        conversation: List[TransformersChainMessage],
        response: str,
    ) -> List[TransformersChainMessage]:
        tool_calls = ChatCausalMultiTurnsChain.extract_json(response)

        for tool_call in tool_calls:
            tool_json = json.loads(tool_call)
            if tool_json["name"] not in self._tool_dict:
                res = f"Tool {tool_json['name']} does not exist"
                conversation.append({"role": "tool", "content": res})
            else:
                try:
                    tool_result = self._tool_dict[tool_json["name"]](
                        **tool_json["arguments"]
                    )
                    conversation.append({"role": "tool", "content": str(tool_result)})
                except Exception as e:
                    res = (
                        f"Encountered error while calling tool {tool_json['name']}. {e}"
                    )
                    conversation.append({"role": "tool", "content": res})
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[TransformersChainMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            "assistant": self.name,
            "user": "user",
            "tool": "tool",
            "system": "system",
        }
        for i in range(start_index, end_index):
            if (
                isinstance(conversation[i]["content"], str)
                and len(conversation[i]["content"]) > 0
            ):
                message.responses.append(
                    (name_map[conversation[i]["role"]], conversation[i]["content"])
                )
        # TODO: handle other modals later
        return message

    @property
    def tools(self) -> List[Callable]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[Callable]):
        self._tool_dict = {tool.__name__: tool for tool in tools}

    @property
    def model(self) -> Tuple[Any, Any]:
        # Return the processor when one was loaded so a read-modify-reassign
        # round-trip through the setter preserves the pixel pipeline.
        formatter = self._processor if self._processor is not None else self._tokenizer
        return formatter, self._model

    @model.setter
    def model(self, model: str | os.PathLike | Tuple[Any, Any]):
        self._bind_model(model)
