import unittest
from unittest import TestCase
from simple_transformer.letter_tokeniser import LetterTokeniser

class TestLetterTokeniser(TestCase):
    def setUp(self) -> None:
        self.tokeniser = LetterTokeniser()

    def test_roundtrip(self) -> None:
        self.assertEqual(
            self.tokeniser.decode(self.tokeniser.encode("Hello world!")),
            "Hello world!"
        )

    def test_encode(self) -> None:
        self.assertEqual(self.tokeniser.encode("Hello world!"), [30, 56, 63, 63, 66, 2, 74, 66, 69, 63, 55, 3])

    def test_decode(self) -> None:
        self.assertEqual(self.tokeniser.decode([30, 56, 63, 63, 66, 2, 74, 66, 69, 63, 55, 3]), "Hello world!")

    def test_encode_unknown_character(self) -> None:
        self.assertEqual(self.tokeniser.encode("🤖"), [0])

    def test_decode_unknown_token(self) -> None:
        self.assertEqual(self.tokeniser.decode([0]), "<unknown>")

    def test_vocab_size(self) -> None:
        self.assertEqual(self.tokeniser.vocab_size(), 78)
