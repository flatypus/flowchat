from ..types import *
from PIL.Image import Image as PILImage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as ChatCompletionChunk
from typing import AsyncGenerator, Callable, Generator

def encode_image(image: PILImage, format_type: str = 'PNG') -> str: ...
def wrap_stream_and_count(generator: StreamCompletion, model: str, callback: Callable[[int, int, str], None], plain_text_stream: bool = False) -> Generator[str | ChatCompletionChunk | None, None, None]: ...
async def async_wrap_stream_and_count(generator: AsyncStreamCompletion, model: str, callback: Callable[[int, int, str], None], plain_text_stream: bool = False) -> AsyncGenerator[str | ChatCompletionChunk | None, None]: ...
