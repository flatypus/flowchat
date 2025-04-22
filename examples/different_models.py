from flowchat import Chain

chain = (
    Chain(model="anthropic:claude-3-5-sonnet-20240620", provider="anthropic")
    .anchor("You are a historian.")
    .link("What is the capital of France?")
    .pull().log().unhook()
    
    .link(lambda desc: f"Extract the city in this statement: {desc}")
    .pull(json_schema={"city": "string"})
    .transform(lambda city_json: city_json["city"])
    .log().unhook()
)

print(f"Result: {chain.last()}")
