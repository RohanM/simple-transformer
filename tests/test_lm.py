import unittest
from unittest import TestCase
from simple_transformer.lm import LM

class TestLM(TestCase):
    def test_query_one_token(self) -> None:
        lm = LM()
        result = lm.query_one_token('Hello, world!')
        self.assertTrue(type(result) == str)
        self.assertTrue(len(result) > 0)

    def test_query(self) -> None:
        lm = LM()
        result = lm.query('Hello, world!', response_len=2)
        self.assertEqual(result, 'hello, world!!!')
