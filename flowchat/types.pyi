from PIL.Image import Image as PILImage
from _typeshed import Incomplete
from datetime import datetime
from openai import AsyncStream, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict

StreamCompletion = Stream[ChatCompletionChunk]
AsyncStreamCompletion = AsyncStream[ChatCompletionChunk]
CreateResponse: Incomplete

class Message(TypedDict):
    role: str
    content: str | list[Any]

class ResponseFormat(TypedDict):
    type: Literal['text', 'json_object']

class ImageFormat(TypedDict):
    url: str | PILImage
    format_type: str
    detail: Literal['low', 'high']

class StreamOptions(TypedDict):
    include_usage: bool

class Function(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[Any]

class Tool(TypedDict):
    type: str
    function: Function

class ToolChoiceFunction(TypedDict):
    name: str

class ToolChoice(TypedDict):
    type: str
    function: ToolChoiceFunction

class RequestParams(TypedDict, total=False):
    model: NotRequired[str]
    frequency_penalty: NotRequired[float | int]
    logit_bias: NotRequired[dict[Any, Any]]
    logprobs: NotRequired[bool]
    top_logprobs: NotRequired[int]
    max_tokens: NotRequired[float | int]
    n: NotRequired[float | int]
    presence_penalty: NotRequired[float | int]
    response_format: NotRequired[ResponseFormat]
    seed: NotRequired[int]
    stop: NotRequired[str | list[str]]
    stream_options: NotRequired[StreamOptions]
    temperature: NotRequired[float | int]
    top_p: NotRequired[float | int]
    tools: NotRequired[list[Tool]]
    tool_choice: NotRequired[str | ToolChoice]
    user: NotRequired[str]

class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int

class DetailedUsage(TypedDict):
    model: str
    usage: Usage
    time: datetime
