# flowchat

A Python library for building clean and efficient prompt chains. It is built on top of [OpenAI's Python API](https://github.com/openai/openai-python).
The library is designed to make multi-step prompt chains easy to build and debug.

## Installation
```bash
pip install flowchat
```

## Usage
```py
chain = (
    FilterChain(model="gpt-3.5-turbo")
    .set_system("You are a historian.")
    .with_user_prompt("What is the capital of France?")
    .get_response().log().reset()

    .transform(lambda desc: f"Extract the city in this statement (one word):\n{desc}")
    .get_response().log()

    .set_system("You are an expert storyteller.")
    .transform(lambda city: f"Design a basic three-act point-form short story about {city}.")
    .get_response(max_tokens=512).log()

    .set_system("You are a novelist. Your job is to write a novel about a story that you have heard.")
    .transform(lambda storyline: f"Briefly elaborate on the first act of the storyline: {storyline}")
    .get_response(max_tokens=256, model="gpt-4-1106-preview").log()

    .transform(lambda act: f"Summarize this act into exactly three words:\n{act}")
    .get_response(model="gpt-4")
    .log_tokens()
)

print(f"FINAL RESULT: {chain.text()}")
