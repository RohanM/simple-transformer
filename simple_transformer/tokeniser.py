from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import words
from typing import cast

class Tokeniser:
    NO_TOKEN = -1
    PUNCTUATION = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\"]

    def __init__(self) -> None:
        self.dictionary = words.words() + self.PUNCTUATION

    def encode(self, text: str) -> list[int]:
        return [self.token_for(w) for w in wordpunct_tokenize(text)]

    def decode(self, tokens: list[int]) -> str:
        return self.join_words([self.dictionary[t] for t in tokens])

    def token_for(self, word: str) -> int:
        try:
            return cast(int, self.dictionary.index(word.lower()))
        except ValueError:
            return self.NO_TOKEN

    def join_words(self, words: list[str]) -> str:
        """Join words with spaces between words, and no spaces before punctuation."""
        joined_words = []
        for i, word in enumerate(words):
            joined_words.append(word)
            if i < len(words) - 1 and not self.is_punctuation(words[i + 1]):
                joined_words.append(" ")
        return ''.join(joined_words)

    def is_punctuation(self, word: str) -> bool:
        return word in self.PUNCTUATION
