from textwrap import dedent


def autodedent(*text_lines) -> str:
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
    text_lines = [i if isinstance(i, str) else str(i) for i in text_lines]
    return dedent('\n'.join(text_lines)).strip("\n")
