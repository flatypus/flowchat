from ..types import *
from io import BytesIO
from PIL.Image import Image
from typing import List, Callable
import base64
import tiktoken


def encode_image(image: Image, format_type: str = "PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"


class CountStreamTokens:
    def __init__(self, messages: List[Message]):
        self.collect_tokens: List[str] = []
        self.messages = messages

    def _count_tokens(self, message: str, model: str):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(message))

    def count(self, generator: StreamChatCompletion, callback: Callable[[int], None]):
        for response in generator:
            content = response.choices[0].delta.content
            yield response

            if content is None:
                output_message = "".join(self.collect_tokens)
                tokens = self._count_tokens(output_message, response.model)
                callback(tokens)
                continue

            self.collect_tokens.append(content)
