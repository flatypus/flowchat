from typing import Callable
from ..types import *
from io import BytesIO
from PIL.Image import Image as PILImage
import base64


def encode_image(image: PILImage, format_type: str = "PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"


def wrap_stream_and_count(generator: StreamChatCompletion, model: str, callback: Callable[[int, int, str], None]):
    for response in generator:
        if len(response.choices) == 0:
            usage = response.usage
            if usage is not None:
                callback(usage.prompt_tokens, usage.completion_tokens, model)
            continue
        yield response
