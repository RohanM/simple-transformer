import unittest
from unittest import TestCase
from simple_transformer.lm import LM

class TestLM(TestCase):
    def test_lm(self) -> None:
        lm = LM()
        self.assertEqual(lm.query('Hello, world!'), 'Nice world.')
