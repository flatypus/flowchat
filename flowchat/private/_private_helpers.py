from io import BytesIO
import base64


def _encode_image(image, format_type="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"
