from .autodedent import autodedent
from .private._private_helpers import encode_image
from PIL.Image import Image as PILImage
from retry import retry
from typing import List, Optional, TypedDict, Union, Callable, Dict, Literal, Any, Generator
from typing_extensions import Unpack, NotRequired
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout
import json
import logging
import openai
import os

logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

Message = TypedDict('Message', {'role': str, 'content': str | List[Any]})
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


class Chain:
    def __init__(self, model: str, api_key: str = "", environ_key: str = "OPENAI_API_KEY") -> None:
        super().__init__()

        if type(model) is not str:
            raise TypeError(
                f"Model argument must be a string, not {type(model)}"
            )

        if api_key != "" and type(api_key) is not str:
            raise TypeError(
                f"API key argument must be a string, not {type(api_key)}"
            )

        if type(environ_key) is not str:
            raise TypeError(
                f"Environment key argument must be a string, not {type(environ_key)}"
            )

        if api_key == "":
            env_api_key = os.environ.get(environ_key)
            if env_api_key is not None:
                api_key = env_api_key

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable, "
                "pass in an api_key parameter, or set the environ_key parameter to the environment "
                "variable that contains your API key."
            )
        openai.api_key = api_key

        self.model = model
        self.system: Message | None = None
        self.user_prompt: List[Message] = []
        self.model_response = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _query_api(self, function: Callable[[Any], Any], *args: Any, max_query_time: int | None = None, **kwargs: Any) -> Any:
        """Call the API for max_query_time seconds, and if it times out, it will retry."""
        timeouted_function = timeout(
            dec_timeout=max_query_time, use_signals=False)(function)
        return timeouted_function(*args, **kwargs)

    def _try_query_and_parse(self, function: Callable[[Any], Any], json_schema: Any, *args: Any, max_query_time: int | None = None, stream: bool = False, **kwargs: Any) -> Any:
        """Query and try to parse the response, and if it fails, it will retry."""
        completion = self._query_api(
            function, *args, max_query_time=max_query_time, stream=stream, **kwargs)

        if completion is None:
            return None

        if stream:
            return completion

        message = completion.choices[0].message.content

        if not json_schema is None:
            open_bracket = message.find('{')
            close_bracket = message.rfind('}')
            message = message[open_bracket:close_bracket+1]
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                raise Exception(
                    "Response was not in the expected JSON format. Please try again. Check that you haven't accidentally lowered the max_tokens parameter so that the response is truncated."
                )

        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens

        return message

    def _ask(
        self,
        system: Message | None,
        user_messages: List[Message],
        json_schema: Optional[dict[Any, Any]] = None,
        max_query_time: Optional[int] = None,
        tries: int = -1,
        stream: bool = False,
        **params: Any
    ) -> Any:
        """Ask a question to the chatbot with a system prompt and return the response."""

        messages = [
            system,
            *user_messages
        ] if system else user_messages

        message = retry(delay=1, logger=logging.getLogger(), tries=tries)(self._try_query_and_parse)(
            openai.chat.completions.create,  # type: ignore
            json_schema=json_schema,
            messages=messages,
            max_query_time=max_query_time,
            stream=stream,
            **params
        )

        return message

    def _format_images(self, image: str | ImageFormat | Any) -> dict[str, str]:
        """Format whatever image format we receive into the specific format that OpenAI's API expects."""
        if isinstance(image, str):
            return {"url": image}
        elif not isinstance(image, dict):
            # not string or dict so assume PIL image
            # no specific file format, so default to PNG
            return {"url": encode_image(image, "PNG")}
        else:
            # we've received an object then; encode the image if necessary
            if 'url' not in image:
                raise Exception(
                    "Image object must have a url property."
                )
            if isinstance(image['url'], str):
                url = image['url']
            else:
                file_format = image['format_type'] if 'format_type' in image else "PNG"
                url = encode_image(image['url'], file_format)

            return {
                "url": url,
                **({"detail": image["detail"]} if "detail" in image else {})
            }

    def unhook(self) -> 'Chain':
        """Reset the chain's system and user prompt. The previous response is kept."""
        self.system = None
        self.user_prompt = []
        return self

    def anchor(self, system_prompt: str) -> 'Chain':
        """Set the chain's system prompt."""

        self.system = {"role": "system", "content": system_prompt}
        return self

    def transform(self, function: Callable[[Any], Any]) -> 'Chain':
        """Transform the chain's model response with a function."""
        if not callable(function):
            raise TypeError(
                f"Transform function must be callable, not {type(function)}"
            )

        self.model_response = (
            None if self.model_response is None else
            function(self.model_response)
        )
        return self

    def link(self, modifier: Union[Callable[[str | Any | None], str], str], model: str | None = None, assistant: bool = False, images: str | Any | List[str | Any] | ImageFormat = None) -> 'Chain':
        """Modify the chain's user prompt with a function, or just pass in a string to be added to the message list.

        For example:
        ```
        chain = (
            Chain()
            .anchor("Hello!")
            .link("How are you?")
            .pull().unhook()

            .link(lambda response: f"What emotions characterize this response? {response}")
            .pull()
            .log()
        )
        ```
        """
        if model is None:
            model = self.model

        if isinstance(modifier, str) and modifier == "":
            raise ValueError(
                "Modifier cannot be an empty string."
            )

        prompt = modifier(self.model_response) if callable(
            modifier) else modifier

        role = "assistant" if assistant else "user"

        if images is None:
            self.user_prompt.append({"role": role, "content": prompt})
        else:
            # images accepts a string (url), a PIL image, as well as a specific typed dict, or a list of any of these
            images = [images] if not isinstance(images, list) else images
            images = [
                {"type": "image_url", "image_url": self._format_images(image)}
                for image in images
            ]
            self.user_prompt.append(
                {"role": role, "content": [
                    {"type": "text", "text": prompt},
                    *images
                ]}
            )
        return self

    def pull(self, json_schema: Optional[dict[Any, Any]] = None, max_query_time: Optional[int] = None, tries: int = -1, **params: Unpack[RequestParams]) -> 'Chain':
        """Makes a request to the LLM and sets the response."""

        params['model'] = params.get('model', self.model)

        if len(self.user_prompt) == 0:
            raise ValueError(
                "User prompt is empty. Please set a user prompt before pulling."
            )

        if json_schema is not None:
            params['response_format'] = {'type': 'json_object'}
            params['model'] = 'gpt-4-turbo'
            self.user_prompt[-1]['content'] += autodedent(
                "You must respond in the following example JSON format. Remember to enclose the entire JSON object in curly braces:",
                json.dumps(json_schema, indent=4)
            )

        response = self._ask(
            self.system, self.user_prompt,
            json_schema=json_schema, max_query_time=max_query_time,
            tries=tries, **params
        )

        self.model_response = response
        return self

    def stream(
        self,
        plain_text_stream: bool = False,
        max_query_time: Optional[int] = None,
        **params: Unpack[RequestParams]

    ) -> Generator[str, None, None]:
        """Returns a generator that yields responses from the LLM."""
        params['model'] = params.get('model', self.model)

        if len(self.user_prompt) == 0:
            raise ValueError(
                "User prompt is empty. Please set a user prompt before pulling."
            )

        return (
            response.choices[0].delta.content if plain_text_stream else response
            for response in self._ask(
                self.system, self.user_prompt,
                json_schema=None, max_query_time=max_query_time,
                stream=True, **params
            )
        )

    def last(self) -> Any:
        """Return the chain's last model response."""
        if self.model_response is None:
            raise ValueError(
                "Model response is empty. Please pull a response before calling last()."
            )
        return self.model_response

    def token_usage(self) -> tuple[int, int]:
        """Return the number of tokens used"""
        return self.prompt_tokens, self.completion_tokens

    def log(self) -> 'Chain':
        """Log the chain's system prompt, user prompt, and model response."""
        print('='*60)
        print(f"System: {self.system}")
        print(f"User: {self.user_prompt}")
        print(f"Text: {self.model_response}")
        print('='*60)
        print("\n")
        return self

    def log_tokens(self) -> 'Chain':
        """Log the number of tokens used"""
        prompt, completion = self.token_usage()
        print(f"Prompt tokens: {prompt}")
        print(f"Completion tokens: {completion}")
        print(f"Total tokens: {prompt + completion}")
        return self
