from openai import Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from PIL.Image import Image as PILImage
from typing import List,  TypedDict, Literal, Any


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
