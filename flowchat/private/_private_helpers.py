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


def wrap_stream_and_count(generator: StreamCompletion, model: str, callback: Callable[[int, int, str], None], plain_text_stream: bool = False):
    for response in generator:
        if len(response.choices) == 0:
            usage = response.usage
            if usage is not None:
                callback(usage.prompt_tokens, usage.completion_tokens, model)
            continue
        content = response.choices[0].delta.content
        yield content if plain_text_stream else response


async def async_wrap_stream_and_count(generator: AsyncStreamCompletion, model: str, callback: Callable[[int, int, str], None], plain_text_stream: bool = False):
    async for response in generator:
        if len(response.choices) == 0:
            usage = response.usage
            if usage is not None:
                callback(usage.prompt_tokens, usage.completion_tokens, model)
            continue
        content = response.choices[0].delta.content
        yield content if plain_text_stream else response
