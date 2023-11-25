from flowchat import Chain
from unittest.mock import patch
import pytest


class TestChaining:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.equal_barrier = "============================================================"

    def test_anchor(self):
        # Test that the anchor method works as expected
        chain = Chain(model="gpt-3.5-turbo").anchor("Hello!")
        assert chain.system == {"role": "system", "content": "Hello!"}

    def test_invalid_anchor(self):
        # Test that the anchor method raises a TypeError when given an invalid argument
        chain = Chain(model="gpt-3.5-turbo")
        with pytest.raises(TypeError):
            chain.anchor(1)

    def test_link(self):
        # Test that the link method works as expected
        chain = Chain(
            model="gpt-3.5-turbo").link("How are you?")
        assert chain.user_prompt == [
            {"role": "user", "content": "How are you?"}
        ]

    def test_invalid_link(self):
        # Test that the link method raises a TypeError when given an invalid argument
        chain = Chain(model="gpt-3.5-turbo")
        with pytest.raises(TypeError):
            chain.link(1)

    def test_link_with_modifier(self):
        with patch('flowchat.Chain._ask') as mock_ask:
            mock_ask.return_value = "Paris"
            chain = (
                Chain(model="gpt-3.5-turbo")
                .link("What is the capital of France?")
                .pull()
                .link(lambda x: f"The capital of France is {x}.")
            )
            mock_ask.assert_called_once()
            assert chain.user_prompt == [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "user", "content": "The capital of France is Paris."},
            ]

    def test_multiple_links(self):
        # Test that the link method works as expected when called multiple times
        chain = (
            Chain(model="gpt-3.5-turbo")
            .link("How are you?")
            .link("What is your name?")
        )
        assert chain.user_prompt == [
            {"role": "user", "content": "How are you?"},
            {"role": "user", "content": "What is your name?"},
        ]

    def test_assistant(self):
        # Test that the assistant role works as expected
        chain = Chain(model="gpt-3.5-turbo").link("Hello!", assistant=True)
        assert chain.user_prompt == [
            {"role": "assistant", "content": "Hello!"}
        ]

    def test_transform(self):
        # Test that the transform method works as expected
        with patch('flowchat.Chain._ask') as mock_ask:
            mock_ask.return_value = "Paris"
            chain = (
                Chain(model="gpt-3.5-turbo")
                .link("What is the capital of France?")
                .pull().transform(lambda x: x.upper())
            )
            mock_ask.assert_called_once()
            assert chain.model_response == "PARIS"

    def test_invalid_transform(self):
        # Test that the transform method raises a TypeError when given an invalid argument
        chain = Chain(model="gpt-3.5-turbo")
        with pytest.raises(TypeError):
            chain.transform(1)

    def test_unhook(self):
        # Test that the reset method works as expected
        chain = (
            Chain(model="gpt-3.5-turbo")
            .link("How are you?")
            .unhook()
            .link("What is your name?")
        )
        assert chain.user_prompt == [
            {"role": "user", "content": "What is your name?"}
        ]

    def test_log(self, capsys):
        # Test that the log method works as expected
        (Chain(model="gpt-3.5-turbo").link("How are you?").log())
        captured = capsys.readouterr()
        assert captured.out == f"{self.equal_barrier}\nSystem: {None}\nUser: {[{'role': 'user', 'content': 'How are you?'}]}\nText: {None}\n{self.equal_barrier}\n\n\n"

    def test_token_usage(self):
        # test that the token_usage method works as expected
        chain = Chain(model="gpt-3.5-turbo")
        chain.prompt_tokens = 1
        chain.completion_tokens = 2
        usage = chain.token_usage()
        assert usage == (1, 2)

    def test_log_tokens(self, capsys):
        # test that the log_tokens method works as expected
        chain = Chain(model="gpt-3.5-turbo")
        chain.prompt_tokens = 1
        chain.completion_tokens = 2
        chain.log_tokens()
        captured = capsys.readouterr()
        assert captured.out == f"Prompt tokens: 1\nCompletion tokens: 2\nTotal tokens: 3\n"
