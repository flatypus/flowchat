import unittest
from unittest.mock import Mock, patch
from flowchat import Chain

# Enum for ResponseFormat is not defined. We define a mock for test purposes


class ResponseFormat:
    JSON = 'json_object'
    TEXT = 'text'


class TestChain(unittest.TestCase):
    def setUp(self):
        self.chain = Chain(model='gpt-3.5-turbo')
        self.chain._ask = Mock(return_value="mocked response")
        self.chain.user_prompt = [{'content': 'Test prompt'}]

    def test_pull_with_default_params(self):
        self.chain.pull()
        self.chain._ask.assert_called_once()

    def test_pull_with_custom_params(self):
        self.chain.pull(model='new-model', max_tokens=100)
        expected_params = {
            'max_tokens': 100,
            'model': 'new-model'
        }
        self.chain._ask.assert_called_with(
            self.chain.system, self.chain.user_prompt,
            None, **expected_params
        )

    def test_pull_with_json_schema(self):
        with patch('json.dumps') as mock_json_dumps:
            mock_json_dumps.return_value = '{}'
            schema = {'type': 'object'}
            (self.chain.link('What is your name?').pull(json_schema=schema))
            mock_json_dumps.assert_called_with(schema, indent=4)
            self.chain.user_prompt[-1]['content'].endswith('{}')

    def test_pull_sets_response(self):
        response = self.chain.pull().model_response
        self.assertEqual(response, "mocked response")

    def test_pull_with_response_format(self):
        self.chain.pull(response_format=ResponseFormat.JSON)
        self.chain._ask.assert_called_once()
        self.assertEqual(self.chain.model_response, "mocked response")


class TestPullParams(unittest.TestCase):
    def setUp(self):
        self.chain = Chain(model='gpt-3.5-turbo')
        self.chain._ask = Mock(return_value="mocked response")

    def test_pull_with_max_query_time(self):
        self.chain.pull(max_query_time=100)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['max_query_time'], 100)

    def test_pull_with_frequency_penalty(self):
        self.chain.pull(frequency_penalty=0.5)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['frequency_penalty'], 0.5)

    def test_pull_with_top_p(self):
        self.chain.pull(top_p=0.9)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['top_p'], 0.9)

    def test_pull_with_logit_bias(self):
        self.chain.pull(logit_bias={'test': 1})
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['logit_bias'], {'test': 1})

    def test_pull_with_presence_penalty(self):
        self.chain.pull(presence_penalty=0.5)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['presence_penalty'], 0.5)

    def test_pull_with_seed(self):
        self.chain.pull(seed=123)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['seed'], 123)

    def test_pull_with_stop(self):
        self.chain.pull(stop=['test'])
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['stop'], ['test'])

    def test_pull_with_temperature(self):
        self.chain.pull(temperature=0.5)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['temperature'], 0.5)

    def test_pull_with_n(self):
        self.chain.pull(n=1)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['n'], 1)

    def test_pull_with_max_tokens(self):
        self.chain.pull(max_tokens=100)
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['max_tokens'], 100)

    def test_pull_with_model(self):
        self.chain.pull(model='new-model')
        self.chain._ask.assert_called_once()
        args = self.chain._ask.call_args
        self.assertEqual(args[1]['model'], 'new-model')

    def test_pull_with_invalid_params(self):
        with self.assertRaises(TypeError):
            self.chain.pull(invalid_param=1)

    def test_pull_with_invalid_schema(self):
        with self.assertRaises(TypeError):
            self.chain.pull(json_schema=1)
