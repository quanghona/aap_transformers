import base64
import io
from typing import Any, Dict

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

    Other modalities (audio, video, document) have their own part types and get
    their own mappers when wired up; this helper refuses non-image refs rather
    than silently mislabeling them.

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
