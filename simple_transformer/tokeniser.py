from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import words

class Tokeniser:
    word_by_idx: list[str]
    idx_by_word: dict[str, int]

    NO_TOKEN = 0
    NO_TOKEN_STR = "<unknown>"
    PUNCTUATION = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\"]

    def __init__(self) -> None:
        self.word_by_idx = [self.NO_TOKEN_STR] + sorted(words.words()) + self.PUNCTUATION
        self.idx_by_word = { w: i for i, w in enumerate(self.word_by_idx) }

    def encode(self, text: str) -> list[int]:
        return [self.token_for(w) for w in wordpunct_tokenize(text)]

    def decode(self, tokens: list[int]) -> str:
        return self.join_words([self.word_by_idx[t] for t in tokens])

    def vocab_size(self) -> int:
        return len(self.word_by_idx)

    def token_for(self, word: str) -> int:
        return self.idx_by_word.get(word.lower(), self.NO_TOKEN)

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
