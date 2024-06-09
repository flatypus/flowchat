import asyncio
from flowchat import Chain


async def main():
    chain = Chain(model="gpt-3.5-turbo")
    await chain.async_pull()

asyncio.run(main())
