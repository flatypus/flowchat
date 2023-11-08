from private._private_helpers import _try_function_until_success
import openai
import os


class FilterChain:
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = os.environ.get("OPENAI_API_KEY")):
        super().__init__()
        if not api_key:
            raise Exception(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = api_key

        self.model = model
        self.system = None
        self.user_prompt = None
        self.model_response = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask_simple(self, system: str, user: str, model: str = None, max_tokens: int = None):
        """Ask a question to the chatbot with a single system prompt and return the response."""
        if model is None:
            model = self.model
        if not user:
            return None
        completion = _try_function_until_success(
            openai.chat.completions.create,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ] if system else [
                {
                    "role": "user",
                    "content": user
                },
            ],
            max_tokens=max_tokens
        )
        if completion is None:
            return None
        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens
        return completion.choices[0].message.content

    def reset(self):
        """Reset the chain's system and user prompt."""
        self.system = None
        self.user_prompt = None
        return self

    def set_system(self, system_prompt: str):
        """Set the chain's system prompt."""
        self.system = system_prompt
        return self

    def with_user_prompt(self, user_prompt: str):
        """Set the chain's user prompt."""
        self.user_prompt = user_prompt
        return self

    def get_response(self, model: str = None, max_tokens: int = None):
        """Make a request to LLM and return the response."""
        if model is None:
            model = self.model
        response = self.ask_simple(
            self.system, self.user_prompt, model=model, max_tokens=max_tokens)
        self.model_response = response
        return self

    def transform(self, transform_function: callable, model: str = None):
        """Modify the chain's user prompt with a function."""
        if model is None:
            model = self.model
        prompt = transform_function(self.model_response)
        self.with_user_prompt(prompt)
        return self

    def text(self) -> str:
        """Return the chain's last model response."""
        return self.model_response

    def token_usage(self) -> int:
        """Return the number of tokens used"""
        return self.prompt_tokens + self.completion_tokens

    def log(self):
        """Log the chain's system prompt, user prompt, and model response."""
        print('='*30)
        print(f"System: {self.system}")
        print(f"User: {self.user_prompt}")
        print(f"Text: {self.model_response}")
        print('='*30)
        print("\n")
        return self

    def log_tokens(self):
        """Log the number of tokens used"""
        print(f"Prompt tokens: {self.prompt_tokens}")
        print(f"Completion tokens: {self.completion_tokens}")
        print(f"Total tokens: {self.token_usage()}")
        return self
