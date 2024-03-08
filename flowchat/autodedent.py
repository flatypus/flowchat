from textwrap import dedent
from typing import List, Any


def autodedent(*text_lines: Any) -> str:
    """Format multiline strings, including with multiple levels of indentation, to align with the first line.

    Example:

    code = '''
    def add(a, b):
        return a + b
    '''

    autodedent(
        "What does this code do?",
        code,
        "Suggest a comment that describes what this code does."
    )
    """

    text_lines_str: List[str] = [i if isinstance(
        i, str) else str(i) for i in text_lines]

    return dedent('\n'.join(text_lines_str)).strip("\n")
