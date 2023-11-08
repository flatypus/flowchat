from contextlib import contextmanager
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
