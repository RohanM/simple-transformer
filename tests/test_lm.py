import unittest
from unittest import TestCase
from simple_transformer.lm import LM

class TestLM(TestCase):
    def test_query(self) -> None:
        lm = LM()
        result = lm.query('zz', response_len=10)
        self.assertEqual(len(result), 12)
