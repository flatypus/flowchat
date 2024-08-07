from .types import *
from .autodedent import autodedent as autodedent
from .private._private_helpers import async_wrap_stream_and_count as async_wrap_stream_and_count, encode_image as encode_image, wrap_stream_and_count as wrap_stream_and_count
from _typeshed import Incomplete
from typing import Any, AsyncGenerator, Callable, Generator
from typing_extensions import Unpack

class Chain:
    client: Incomplete
    asyncClient: Incomplete
    model: Incomplete
    system: Incomplete
    user_prompt: Incomplete
    model_response: Incomplete
    raw_model_response: Incomplete
    usage: Incomplete
    detailed_usage: Incomplete
    def __init__(self, model: str, api_key: str = '', environ_key: str = 'OPENAI_API_KEY') -> None: ...
    async def async_ask(self, system: Message | None, user_messages: list[Message], json_schema: dict[Any, Any] | None = None, stream: bool = False, plain_text_stream: bool = False, **params: Unpack[RequestParams]) -> Any: ...
    def unhook(self) -> Chain: ...
    def anchor(self, system_prompt: str) -> Chain: ...
    def transform(self, function: Callable[[Any], Any]) -> Chain: ...
    def link(self, modifier: Callable[[str | Any | None], str] | str, model: str | None = None, assistant: bool = False, images: str | Any | list[str | Any] | ImageFormat = None) -> Chain: ...
    def setup_pull(self, asynchronous: bool = False, json_schema: dict[Any, Any] | None = None, **params: Unpack[RequestParams]) -> Any: ...
    def pull(self, json_schema: dict[Any, Any] | None = None, **params: Unpack[RequestParams]) -> Chain: ...
    async def async_pull(self, json_schema: dict[Any, Any] | None = None, **params: Unpack[RequestParams]) -> Chain: ...
    def stream(self, plain_text_stream: bool = False, **params: Unpack[RequestParams]) -> Generator[str | Any | None, None, None]: ...
    async def async_stream(self, plain_text_stream: bool = False, **params: Unpack[RequestParams]) -> AsyncGenerator[str | Any | None, None]: ...
    def last(self) -> Any: ...
    def token_usage(self) -> Usage: ...
    def detailed_token_usage(self) -> list[DetailedUsage]: ...
    def reset_token_usage(self) -> Chain: ...
    def subscribe_token_usage(self, usage_object: Usage) -> Chain: ...
    def subscribe_detailed_token_usage(self, detailed_usage_object: list[DetailedUsage]) -> Chain: ...
    def log(self) -> Chain: ...
    def log_tokens(self) -> Chain: ...
    def log_detailed_tokens(self) -> Chain: ...
