import unittest
from unittest import TestCase
from simple_transformer.lm import LM

class TestLM(TestCase):
    def test_query(self) -> None:
        lm = LM()
        result = lm.query('zz', response_len=10)
        self.assertEqual(len(result), 12)

    def test_short_query(self) -> None:
        lm = LM()
        result = lm.query('z', response_len=10)
        self.assertEqual(len(result), 11)

    def test_empty_query(self) -> None:
        lm = LM()
        with self.assertRaises(ValueError):
            result = lm.query('', response_len=10)
