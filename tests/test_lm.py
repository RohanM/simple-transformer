import unittest
from unittest import TestCase
from simple_transformer.lm import LM

class TestLM(TestCase):
    def test_lm(self) -> None:
        lm = LM()
        result = lm.query('Hello, world!')
        self.assertTrue(type(result) == str)
        self.assertTrue(len(result) > 0)
