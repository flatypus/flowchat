import asyncio
from flowchat import Chain


async def main():
    chain = (
        Chain(model="gpt-3.5-turbo")
        .anchor("You are a mathematician.")
        .link("What is the square root of 16?")
    )
    response = ""
    async for token in await chain.async_stream(plain_text_stream=True):
        if token == None:
            continue
        response += token

    print(response)
    chain.log_detailed_tokens()


asyncio.run(main())
