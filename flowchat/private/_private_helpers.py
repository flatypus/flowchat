from ..types import *
from io import BytesIO
from math import ceil
from PIL import Image
from PIL.Image import Image as PILImage
from requests import get
from typing import Callable, List, Dict
import base64
import tiktoken


def encode_image(image: PILImage, format_type: str = "PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"


class CalculateImageTokens:
    def __init__(self, image: str):
        self.image = image

    def _get_image_dimensions(self):
        if self.image.startswith("data:image"):
            image = self.image.split(",")[1]
            image = base64.b64decode(image)
            image = Image.open(BytesIO(image))
            return image.size
        else:
            response = get(self.image)
            image = Image.open(BytesIO(response.content))
            return image.size

    def _openai_resize(self, width: int, height: int):
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024
        return width, height

    def count_image_tokens(self):
        width, height = self._get_image_dimensions()
        width, height = self._openai_resize(width, height)
        h = ceil(height / 512)
        w = ceil(width / 512)
        total = 85 + 170 * h * w
        return total


class CountStreamTokens:
    def __init__(self, model: str, messages: List[Message]):
        self.collect_tokens: List[str] = []
        self.messages = messages
        self.model = self._get_model(model)
        self.tokens_per_message = 3
        self.tokens_per_name = 1

    def _get_model(self, model: str):
        """Picks the right model and sets the additional tokens. See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            self.tokens_per_message = 3
            self.tokens_per_name = 1

        elif model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            self.tokens_per_message = 4
            self.tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            self._get_model("gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            self._get_model("gpt-4-0613")

    def _count_text_tokens(self, message: Message) -> int:
        """Return the number of tokens used by a list of messages. See above link for context"""
        num_tokens = self.tokens_per_message
        for key, value in message.items():
            num_tokens += len(self.encoding.encode(str(value)))
            if key == "name":
                num_tokens += self.tokens_per_name

        return num_tokens

    def _count_input_tokens(self):
        tokens = 0
        text_messages: List[Message] = []
        image_messages: List[Dict[str, Any]] = []

        for message in self.messages:
            content = message["content"]
            role = message["role"]
            if isinstance(content, str):
                text_messages.append({"role": role, "content": content})
            else:
                for item in content:
                    if item["type"] == "text":
                        text_messages.append(
                            {"role": role, "content": item["text"]})
                    else:
                        image_messages.append(item)

        for message in text_messages:
            tokens += self._count_text_tokens(message)

        for message in image_messages:
            image = message["image_url"]
            detail = image.get("detail", "high")
            if detail == "low":
                tokens += 85
            else:
                tokens += (
                    CalculateImageTokens(message["image_url"]["url"])
                    .count_image_tokens()
                )

        tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return tokens

    def _count_output_tokens(self, message: str):
        return len(self.encoding.encode(message))

    def wrap_stream_and_count(self, generator: StreamChatCompletion, callback: Callable[[int, int], None]):
        for response in generator:
            content = response.choices[0].delta.content
            yield response

            if content is None:
                output_message = "".join(self.collect_tokens)
                prompt_tokens = self._count_input_tokens()
                completion_tokens = self._count_output_tokens(output_message)
                callback(prompt_tokens, completion_tokens)
                continue

            self.collect_tokens.append(content)
