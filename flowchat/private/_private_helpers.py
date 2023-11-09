from contextlib import contextmanager
from io import BytesIO
import base64
import signal


class TimeoutException(Exception):
    pass


@contextmanager
def _time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _try_function_until_success(function: callable, timeout: int = 20, *args, **kwargs):
    try:
        with _time_limit(timeout):
            return function(*args, **kwargs)
    except TimeoutException:
        return _try_function_until_success(function, timeout, *args, **kwargs)


def _encode_image(image, format_type="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"
