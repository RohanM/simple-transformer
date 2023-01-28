class LetterTokeniser:
    letter_by_idx: list[str]
    idx_by_letter: dict[str, int]

    NO_TOKEN = 0
    NO_TOKEN_STR = "<unknown>"

    VOCAB = ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '6', '7', '8', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def __init__(self) -> None:
        self.letter_by_idx = [self.NO_TOKEN_STR] + self.VOCAB
        self.idx_by_letter = { l: i for i, l in enumerate(self.letter_by_idx) }

    def encode(self, text: str) -> list[int]:
        return [self.token_for(l) for l in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.letter_by_idx[t] for t in tokens])

    def vocab_size(self) -> int:
        return len(self.letter_by_idx)

    def token_for(self, letter: str) -> int:
        return self.idx_by_letter.get(letter, self.NO_TOKEN)
