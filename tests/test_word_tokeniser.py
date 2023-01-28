import unittest
from unittest import TestCase
from simple_transformer.word_tokeniser import WordTokeniser

class TestWordTokeniser(TestCase):
    def setUp(self) -> None:
        self.tokeniser = WordTokeniser()

    def test_roundtrip(self) -> None:
        self.assertEqual(
            self.tokeniser.decode(self.tokeniser.encode("Hello world!")),
            "hello world!"
        )

    def test_encode(self) -> None:
        self.assertEqual(self.tokeniser.encode("Hello world!"), [99079, 234773, 236739])

    def test_decode(self) -> None:
        self.assertEqual(self.tokeniser.decode([99079, 234773, 236739]), "hello world!")

    def test_encode_unknown_token(self) -> None:
        self.assertEqual(self.tokeniser.encode("xyzzy"), [0])

    def test_decode_unknown_token(self) -> None:
        self.assertEqual(self.tokeniser.decode([0]), "<unknown>")

    def test_vocab_size(self) -> None:
        self.assertEqual(self.tokeniser.vocab_size(), 236755)

    def test_join_words(self) -> None:
        self.assertEqual(self.tokeniser.join_words(["hello", "world", "!"]), "hello world!")
