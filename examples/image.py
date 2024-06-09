from flowchat import Chain, autodedent
from PIL import Image

# ========================================================================== #
# Image from URL

ted_spread_image = "https://upload.wikimedia.org/wikipedia/commons/9/92/TED_Spread.png"

lecture_notes = (
    Chain(model="gpt-4-turbo")
    .anchor(autodedent(
            "As a knowledgeable chatbot specializing in written summaries, provide a detailed explanation based on the given text about a topic.",
            "Your task is to convey the information and insights as a teacher would, in short yet detailed sentence lecture notes."
            ))
    .link("Explain this image.", images=ted_spread_image)
    .pull(max_tokens=512)

    # now given observations from image, create a set of lecture notes without needing to reference the image
    .anchor("You are a teacher. Given observations from an image, create a set of concise lecture notes without needing to reference the image.")
    .link(lambda observations: autodedent(
        "Observations: ",
        observations,
    )).pull(max_tokens=256)
)

print(lecture_notes.last())
lecture_notes.log_detailed_tokens()

# ========================================================================== #
# Image from local file
naruto_image = Image.open("examples/images/naruto.png")

character_description = (
    Chain(model="gpt-4-turbo")
    .link("Who is this?", images={"url": naruto_image, "format_type": "PNG", "detail": "low"})
    .pull(max_tokens=128)
)

print(character_description.last())
character_description.log_detailed_tokens()

# ========================================================================== #
# Multiple Images!

ibm_computer_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/IBM_5100_-_MfK_Bern.jpg/2560px-IBM_5100_-_MfK_Bern.jpg"
gaming_computer_image = "https://upload.wikimedia.org/wikipedia/commons/4/42/Alienware.JPG"

computer_description = (
    Chain(model="gpt-4-turbo")
    .link("What are the differences between these two computers?", images=[ibm_computer_image, gaming_computer_image])
    .pull(max_tokens=256)
)

print(computer_description.last())
computer_description.log_detailed_tokens()
