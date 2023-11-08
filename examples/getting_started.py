from flowchat import Chain

chain = (
    Chain(model="gpt-3.5-turbo")  # default model for all pull() calls
    .anchor("You are a historian.")  # Set the first system prompt
    .link("What is the capital of France?")
    .pull().log().unhook()  # Pull the response, log it, and reset prompts

    .link(lambda desc: f"Extract the city in this statement (one word):\n{desc}")
    .pull().log().unhook()

    .anchor("You are an expert storyteller.")
    .link(lambda city: f"Design a basic three-act point-form short story about {city}.")
    .link("How long should it be?", assistant=True)
    .link("Around 100 words.")  # Example to show multiple links
    .pull(max_tokens=512).log().unhook()

    .anchor("You are a novelist. Your job is to write a novel about a story that you have heard.")
    .link(lambda storyline: f"Briefly elaborate on the first act of the storyline: {storyline}")
    .pull(max_tokens=256, model="gpt-4-1106-preview").log().unhook()

    .link(lambda act: f"Summarize this act in around three words:\n{act}")
    .pull(model="gpt-4")
    .log_tokens()  # Log token usage of the whole chain
)

print(f"Result: {chain.last()}")  # Print the last response
