from .autodedent import autodedent
from .private._private_helpers import encode_image, wrap_stream_and_count, async_wrap_stream_and_count
from .types import *
from datetime import datetime
from typing import List, Optional, Union, Callable, Any, Generator, AsyncGenerator
from typing_extensions import Unpack
import json
import logging
import openai
import os


logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)


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
        self.client = openai.Client(api_key=api_key)
        self.asyncClient = openai.AsyncClient(api_key=api_key)
        self.model = model
        self.system: Message | None = None
        self.user_prompt: List[Message] = []
        self.model_response = None
        self.raw_model_response = None
        self.usage: Usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.detailed_usage: List[DetailedUsage] = []

    def _add_token_count(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Add token counts to the chain's total token count."""
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.detailed_usage.append({
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "time": datetime.now()
        })

    def _get_completion(self, **params: Any) -> CreateResponse:
        """Get a completion from OpenAI's API."""
        return self.client.chat.completions.create(**params)  # type: ignore

    async def _get_async_completion(self, **params: Any) -> CreateResponse:
        """Get a completion from OpenAI's API asynchronously."""
        return await self.asyncClient.chat.completions.create(**params)  # type: ignore

    def _post_completion(self, completion: CreateResponse, model: str, json_schema: Optional[dict[Any, Any]] = None) -> Any:
        if isinstance(completion, ChatCompletion):
            self.raw_model_response = completion
            message = completion.choices[0].message.content
            if message is None:
                raise Exception("Response was empty. Please try again.")

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

            if completion.usage is not None:
                self._add_token_count(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    model
                )

            return message

    def _ask(
        self,
        system: Message | None,
        user_messages: List[Message],
        json_schema: Optional[dict[Any, Any]] = None,
        stream: bool = False,
        plain_text_stream: bool = False,
        **params: Unpack[RequestParams]
    ) -> Any:
        """Ask a question to the chatbot with a system prompt and return the response."""
        model = params.get("model", self.model)

        messages = [
            system,
            *user_messages
        ] if system else user_messages

        completion = self._get_completion(
            messages=messages, stream=stream, **params)

        if completion is None:
            return None

        if stream and isinstance(completion, Stream):
            return wrap_stream_and_count(completion, model, self._add_token_count, plain_text_stream)

        return self._post_completion(completion, model, json_schema)

    async def async_ask(
        self,
        system: Message | None,
        user_messages: List[Message],
        json_schema: Optional[dict[Any, Any]] = None,
        stream: bool = False,
        plain_text_stream: bool = False,
        **params: Unpack[RequestParams]
    ) -> Any:
        """Ask a question to the chatbot with a system prompt and return the response."""
        model = params.get("model", self.model)

        messages = [
            system,
            *user_messages
        ] if system else user_messages

        completion = await self._get_async_completion(messages=messages, stream=stream, **params)

        if completion is None:
            return None

        if stream and isinstance(completion, AsyncStream):
            return async_wrap_stream_and_count(completion, model, self._add_token_count, plain_text_stream)

        return self._post_completion(completion, model, json_schema)

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

    def setup_pull(self, asynchronous: bool = False, json_schema: Optional[dict[Any, Any]] = None, **params: Unpack[RequestParams]):
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

        ask = self.async_ask if asynchronous else self._ask
        return ask(
            self.system, self.user_prompt,
            json_schema=json_schema, **params
        )

    def pull(self, json_schema: Optional[dict[Any, Any]] = None, **params: Unpack[RequestParams]) -> 'Chain':
        """Makes a request to the LLM and sets the response."""
        self.model_response = self.setup_pull(False, json_schema, **params)
        return self

    async def async_pull(self, json_schema: Optional[dict[Any, Any]] = None, **params: Unpack[RequestParams]) -> 'Chain':
        """Makes a request to the LLM and sets the response."""
        self.model_response = await self.setup_pull(True, json_schema, **params)
        return self

    def stream(
        self, plain_text_stream: bool = False,
        **params: Unpack[RequestParams]
    ) -> Generator[str | Any | None, None, None]:
        """Returns a generator that yields responses from the LLM."""
        params['model'] = params.get('model', self.model)
        params['stream_options'] = {'include_usage': True}

        if len(self.user_prompt) == 0:
            raise ValueError(
                "User prompt is empty. Please set a user prompt before pulling."
            )

        return (
            response.choices[0].delta.content if plain_text_stream else response
            for response in self._ask(
                self.system, self.user_prompt,
                json_schema=None,
                stream=True, **params
            )
        )

    async def async_stream(
        self, plain_text_stream: bool = False,
        **params: Unpack[RequestParams]
    ) -> AsyncGenerator[str | Any | None, None]:
        """Returns a generator that yields responses from the LLM."""
        params['model'] = params.get('model', self.model)
        params['stream_options'] = {'include_usage': True}

        if len(self.user_prompt) == 0:
            raise ValueError(
                "User prompt is empty. Please set a user prompt before pulling."
            )

        return (
            response.choices[0].delta.content if plain_text_stream else response
            async for response in await self.async_ask(
                self.system, self.user_prompt,
                json_schema=None,
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

    def token_usage(self) -> Usage:
        """Return the number of tokens used"""
        return self.usage

    def detailed_token_usage(self) -> List[DetailedUsage]:
        """Return the detailed token usage"""
        return self.detailed_usage

    def reset_token_usage(self) -> 'Chain':
        """Reset the token usage"""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.detailed_usage = []
        return self

    def subscribe_token_usage(self, usage_object: Usage) -> 'Chain':
        """Subscribe to a token usage object"""
        self.usage = usage_object
        return self

    def subscribe_detailed_token_usage(self, detailed_usage_object: List[DetailedUsage]) -> 'Chain':
        """Subscribe to a detailed token usage object"""
        self.detailed_usage = detailed_usage_object
        return self

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
        usage = self.token_usage()
        prompt = usage["prompt_tokens"]
        completion = usage["completion_tokens"]
        print(f"Prompt tokens: {prompt}")
        print(f"Completion tokens: {completion}")
        print(f"Total tokens: {prompt + completion}")
        return self

    def log_detailed_tokens(self) -> 'Chain':
        """Log the detailed token usage"""
        for usage in self.detailed_token_usage():
            print('='*60)
            print(f"Model: {usage['model']}")
            print(f"Prompt tokens: {usage['usage']['prompt_tokens']}")
            print(f"Completion tokens: {usage['usage']['completion_tokens']}")
            print(f"Time: {usage['time']}")
            print('='*60)
            print("\n")
        return self
