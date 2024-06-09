from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream, AsyncStream
from PIL.Image import Image as PILImage
from typing import List, NotRequired,  TypedDict, Literal, Any, Union, Dict
from datetime import datetime


StreamCompletion = Stream[ChatCompletionChunk]
AsyncStreamCompletion = AsyncStream[ChatCompletionChunk]
CreateResponse = None | ChatCompletion | StreamCompletion | AsyncStreamCompletion

Message = TypedDict(
    'Message', {'role': str, 'content': str | List[Any]}
)
ResponseFormat = TypedDict(
    'ResponseFormat', {'type': Literal['text', 'json_object']})
ImageFormat = TypedDict('ImageFormat', {
    'url': str | PILImage,
    'format_type': str,
    'detail': Literal['low', 'high']
})

StreamOptions = TypedDict('StreamOptions', {
    'include_usage': bool
})

Function = TypedDict('Function', {
    'name': str,
    'description': NotRequired[str],
    'parameters': NotRequired[Any],
})

Tool = TypedDict('Tool', {
    'type': str,
    'function': Function
})

ToolChoiceFunction = TypedDict('ToolChoiceFunction', {
    'name': str
})

ToolChoice = TypedDict('ToolChoice', {
    'type': str,
    'function': ToolChoiceFunction
})


class RequestParams(TypedDict, total=False):
    model: NotRequired[str]
    frequency_penalty: NotRequired[Union[float, int]]
    logit_bias: NotRequired[Dict[Any, Any]]
    logprobs: NotRequired[bool]
    top_logprobs: NotRequired[int]
    max_tokens: NotRequired[Union[float, int]]
    n: NotRequired[Union[float, int]]
    presence_penalty: NotRequired[Union[float, int]]
    response_format: NotRequired[ResponseFormat]
    seed: NotRequired[int]
    stop: NotRequired[Union[str, List[str]]]
    stream_options: NotRequired[StreamOptions]
    temperature: NotRequired[Union[float, int]]
    top_p: NotRequired[Union[float, int]]
    tools: NotRequired[List[Tool]]
    tool_choice: NotRequired[Union[str, ToolChoice]]
    user: NotRequired[str]


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class DetailedUsage(TypedDict):
    model: str
    usage: Usage
    time: datetime
