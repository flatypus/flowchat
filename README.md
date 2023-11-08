# flowchat

A Python library for building clean and efficient multi-step prompt chains. It is built on top of [OpenAI's Python API](https://github.com/openai/openai-python).

Flowchat is designed around the idea of a *chain*. Each chain can start with a system prompt `.anchor()`, and then add chain links of messages with `.link()`. 

Once a chain has been built, a response from the LLM can be pulled with `.pull()`.

You can optionally log the chain's messages and responses with `.log()`. This is useful for debugging and understanding the chain's behavior. Remember to call `.log()` before `.unhook()` though! Unhooking resets the current chat conversation of the chain.

However, the thing that makes flowchat stand out is the idea of chaining together responses, one chain after another. The chain's previous response can be accessed in the next chain with a lambda function in the next `.link()`. This allows for a more natural conversation flow, and allows for more complex conversations to be built. See examples for how you can use this as well!

## Installation
```bash
pip install flowchat
```

## Usage
```py
chain = (
    Chain(model="gpt-3.5-turbo")
    .anchor("You are a historian.")
    .link("What is the capital of France?")
    .pull().log().unhook()

    .link(lambda desc: f"Extract the city in this statement (one word):\n{desc}")
    .pull().log().unhook()

    .anchor("You are an expert storyteller.")
    .link(lambda city: f"Design a basic three-act point-form short story about {city}.")
    .pull(max_tokens=512).log().unhook()

    .anchor("You are a novelist. Your job is to write a novel about a story that you have heard.")
    .link(lambda storyline: f"Briefly elaborate on the first act of the storyline: {storyline}")
    .pull(max_tokens=256, model="gpt-4-1106-preview").log().unhook()

    .link(lambda act: f"Summarize this act in around three words:\n{act}")
    .pull(model="gpt-4")
    .log_tokens()
)

print(f"Result: {chain.last()}") # >> "Artist's Dream Ignites"

```

This project is under a MIT license.