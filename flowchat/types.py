from openai import Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from PIL.Image import Image as PILImage
from typing import List, NotRequired,  TypedDict, Literal, Any, Union, Dict
from datetime import datetime


StreamChatCompletion = Stream[ChatCompletionChunk]
CreateResponse = None | ChatCompletion | StreamChatCompletion


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

# use total=False to make fields non-required


class RequestParams(TypedDict, total=False):
    model: NotRequired[str]
    frequency_penalty: NotRequired[Union[float, int]]
    logit_bias: NotRequired[Dict[str, Union[float, int]]]
    max_tokens: NotRequired[Union[float, int]]
    n: NotRequired[Union[float, int]]
    presence_penalty: NotRequired[Union[float, int]]
    response_format: NotRequired[ResponseFormat]
    seed: NotRequired[int]
    stop: NotRequired[Union[str, List[str]]]
    temperature: NotRequired[Union[float, int]]
    top_p: NotRequired[Union[float, int]]


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class DetailedUsage(TypedDict):
    model: str
    usage: Usage
    time: datetime
