from flowchat import Chain

# plain text stream allows you to stream the response as chunks of plain text
# you'd probably want stream as the last step in your chain

generator = Chain(model="gpt-3.5-turbo").link("What is the capital of France?")

for response in generator.stream(plain_text_stream=True):
    print(response)

generator.log_detailed_tokens()
