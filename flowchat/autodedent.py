from textwrap import dedent


def autodedent(text):
    return dedent(text).strip("\n")
