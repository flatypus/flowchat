import unittest
from unittest.mock import Mock
from flowchat import Chain
import json


class obj:
    def __init__(self, d):
        self.__dict__.update(d)


class TestStream(unittest.TestCase):
    def dict2obj(self, d):
        return json.loads(json.dumps(d), object_hook=obj)

    def generator(self):
        yield from (self.dict2obj({"choices": [{"delta": {"content": ""}}]}) for _ in range(10))

    def setUp(self):
        self.chain = Chain(model='gpt-3.5-turbo')
        self.chain._ask = Mock(return_value=self.generator())
        self.chain.user_prompt = [{'content': 'Test prompt'}]

    def test_stream(self):
        generator = self.chain.stream()
        for _ in range(10):
            next(generator)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['stream'], True)

    def test_stream_with_plain_text_stream(self):
        generator = self.chain.stream(plain_text_stream=True)
        for _ in range(10):
            item = next(generator)
            assert isinstance(item, str)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['stream'], True)
