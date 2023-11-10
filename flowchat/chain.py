from .autodedent import autodedent
from .private._private_helpers import _try_function_until_success, _encode_image
from typing import List, TypedDict, Union, Callable, Dict, Literal, Any
import json
import openai
import os

Message = TypedDict('Message', {'role': str, 'content': str | List[Any]})
ResponseFormat = TypedDict(
    'ResponseFormat', {'type': Literal['text', 'json_object']})
ImageFormat = TypedDict('ImageFormat', {
    'url': str,
    'format_type': str,
    'detail': Literal['low', 'high']
})


class Chain:
    def __init__(self, model: str, api_key: str = None, environ_key="OPENAI_API_KEY"):
        super().__init__()

        if api_key is None:
            api_key = os.environ.get(environ_key)
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable, "
                "pass in an api_key parameter, or set the environ_key parameter to the environment "
                "variable that contains your API key."
            )
        openai.api_key = api_key

        self.model = model
        self.system = None
        self.user_prompt = []
        self.model_response = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _ask(
        self,
        system: Message,
        user_messages: List[Message],
        **params
    ):
        """Ask a question to the chatbot with a system prompt and return the response."""
        if not user_messages:
            return None

        messages = [
            system,
            *user_messages
        ] if system else user_messages

        completion = _try_function_until_success(
            openai.chat.completions.create,
            messages=messages,
            **params
        )

        if completion is None:
            return None

        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens
        return completion.choices[0].message.content

    def _format_images(self, image: str | ImageFormat | Any):
        """Format whatever image format we receive into the specific format that OpenAI's API expects."""
        if isinstance(image, str):
            return {"url": image}
        elif not isinstance(image, dict):
            # not string or dict so assume PIL image
            # no specific file format, so default to PNG
            return {"url": _encode_image(image, "PNG")}
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
                url = _encode_image(image['url'], file_format)

            return {
                "url": url,
                **({"detail": image["detail"]} if "detail" in image else {})
            }

    def unhook(self):
        """Reset the chain's system and user prompt. The previous response is kept."""
        self.system = None
        self.user_prompt = []
        return self

    def anchor(self, system_prompt: str):
        """Set the chain's system prompt."""
        self.system = {"role": "system", "content": system_prompt}
        return self

    def transform(self, function: Callable[[str], str]):
        """Transform the chain's model response with a function."""
        self.model_response = function(self.model_response)
        return self

    def link(self, modifier: Union[Callable[[str], None], str] = None, model: str = None, assistant=False, images: str | Any | List[str | Any] | ImageFormat = None):
        """Modify the chain's user prompt with a function, or just pass in a string to be added to the message list.

        For example:
        ```
        chain = (Chain()
                    .anchor("Hello!")
                    .link("How are you?")
                    .pull().unhook()

                    .link(lambda response: f"What emotions characterize this response? {response}")
                    .pull()
                    .log())
        ```
        """
        if model is None:
            model = self.model

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

    def pull(
        self,
        model: str = None,
        frequency_penalty: float | int = None,
        json_schema: Any = None,
        logit_bias: Dict[str, float | int] = None,
        max_tokens: float | int = None,
        n: float | int = None,
        presence_penalty: float | int = None,
        response_format: ResponseFormat = None,
        seed: int = None,
        stop: str | List[str] = None,
        temperature: float | int = None,
        top_p: float | int = None
    ):
        """Make a request to the LLM and set the response."""
        if model is None:
            model = self.model

        params = {
            'frequency_penalty': frequency_penalty,
            'logit_bias': logit_bias,
            'max_tokens': max_tokens,
            'model': model,
            'n': n,
            'presence_penalty': presence_penalty,
            'response_format': response_format,
            'seed': seed,
            'stop': stop,
            'temperature': temperature,
            'top_p': top_p,
        }

        params = {k: v for k, v in params.items() if v is not None}

        if json_schema is not None:
            params['response_format'] = {'type': 'json_object'}
            params['model'] = 'gpt-4-1106-preview'
            self.user_prompt[-1]['content'] += autodedent(
                "You must respond in the following example JSON format. Remember to enclose the entire JSON object in curly braces:",
                json.dumps(json_schema, indent=4)
            )

        response = self._ask(self.system, self.user_prompt, **params)

        if json_schema is not None:
            open_bracket = response.rfind('{')
            close_bracket = response.rfind('}')
            response = response[open_bracket:close_bracket+1]
            response = json.loads(response)
        self.model_response = response
        return self

    def last(self) -> str:
        """Return the chain's last model response."""
        return self.model_response

    def token_usage(self) -> int:
        """Return the number of tokens used"""
        return self.prompt_tokens, self.completion_tokens

    def log(self):
        """Log the chain's system prompt, user prompt, and model response."""
        print('='*60)
        print(f"System: {self.system}")
        print(f"User: {self.user_prompt}")
        print(f"Text: {self.model_response}")
        print('='*60)
        print("\n")
        return self

    def log_tokens(self):
        """Log the number of tokens used"""
        prompt, completion = self.token_usage()
        print(f"Prompt tokens: {prompt}")
        print(f"Completion tokens: {completion}")
        print(f"Total tokens: {prompt + completion}")
        return self
