from io import BytesIO
from PIL.Image import Image
import base64


def encode_image(image: Image, format_type: str = "PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"
