from flowchat import autodedent


class TestAutodedent:
    def test_autodedent(self):
        # Test base case
        test_string = """
        This is a test string
        """
        assert autodedent(test_string) == "This is a test string"

    def test_autodedent_with_indent(self):
        # Test with indent
        test_string = """
            This is a test string
        """
        assert autodedent(test_string) == "This is a test string"

    def test_autodedent_with_newline(self):
        # Test with newline
        test_string = """
            This is a test string

        """
        assert autodedent(test_string) == "This is a test string"

    def test_autodedent_with_multiple_lines(self):
        # Test with multiple lines
        test_string = """
            This is a test string
            This is a test string
        """
        assert autodedent(
            test_string) == "This is a test string\nThis is a test string"

    def test_multiple_arguments(self):
        # Test with multiple arguments
        assert autodedent("A", "B") == "A\nB"

    def test_multiple_arguments_with_indent(self):
        # Test with multiple arguments and indent
        assert autodedent("   A", "   B" == "A\nB")

    def test_against_multiple_levels_of_indent(self):
        # Test against multiple levels of indent
        test_string = """
            This is a test string
                This is a test string
        """
        assert autodedent(
            test_string) == "This is a test string\n    This is a test string"

    def test_with_empty_string(self):
        # Test with empty string
        test_string = ""
        assert autodedent(test_string) == ""

    def test_with_non_string(self):
        # Test with non-string
        test_string = 1
        assert autodedent(test_string) == "1"

    def test_with_multiple_non_strings(self):
        # Test with multiple non-strings
        test_string = 1
        test_string2 = 2
        assert autodedent(test_string, test_string2) == "1\n2"
