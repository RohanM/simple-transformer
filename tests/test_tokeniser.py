import unittest
from unittest import TestCase
from simple_transformer.tokeniser import Tokeniser

class TestTokeniser(TestCase):
    def setUp(self) -> None:
        self.tokeniser = Tokeniser()

    def test_roundtrip(self) -> None:
        self.assertEqual(
            self.tokeniser.decode(self.tokeniser.encode("Hello world!")),
            "hello world!"
        )

    def test_encode(self) -> None:
        self.assertEqual(self.tokeniser.encode("Hello world!"), [83713, 233449, 236738])

    def test_decode(self) -> None:
        self.assertEqual(self.tokeniser.decode([83713, 233449, 236738]), "hello world!")

    def test_encode_unknown_token(self) -> None:
        self.assertEqual(self.tokeniser.encode("xyzzy"), [-1])

    def test_vocab_size(self) -> None:
        self.assertEqual(self.tokeniser.vocab_size(), 236754)

    def test_join_words(self) -> None:
        self.assertEqual(self.tokeniser.join_words(["hello", "world", "!"]), "hello world!")
