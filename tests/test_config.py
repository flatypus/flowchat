from flowchat import Chain
import pytest


class TestConfig:
    def test_chain_without_model(self):
        # Test that a TypeError is raised when a Chain is created without a model argument
        with pytest.raises(TypeError):
            Chain()

    def test_chain_with_invalid_model(self):
        # Test that a TypeError is raised when a Chain is created with an invalid model argument
        with pytest.raises(TypeError):
            Chain(model=1)

    def test_chain_with_invalid_api_key(self):
        # Test that a TypeError is raised when a Chain is created with an invalid api_key argument
        with pytest.raises(TypeError):
            Chain(model="gpt-3.5-turbo", api_key=1)

    def test_chain_with_invalid_environ_key(self):
        # Test that a TypeError is raised when a Chain is created with an invalid environ_key argument
        with pytest.raises(TypeError):
            Chain(model="gpt-3.5-turbo", environ_key=1)

    def test_chain_passes(self):
        # Test that a Chain is created successfully
        chain = Chain(model="gpt-3.5-turbo")
        assert isinstance(chain, Chain)

    def test_chain_with_api_key_passes(self):
        # Test that a Chain is created successfully with an api_key argument
        chain = Chain(model="gpt-3.5-turbo", api_key="test")
        assert isinstance(chain, Chain)
