import nltk
from torch import tensor
from torch.utils.data import Dataset
from simple_transformer.tokeniser import Tokeniser
from typing import Iterator

class Data(Dataset[tuple[list[int], list[int]]]):
    windows: list[list[int]]

    def __init__(self, text: str, window_size: int) -> None:
        tokens = Tokeniser().encode(text)
        self.windows = [tensor(ngram) for ngram in nltk.ngrams(tokens, window_size)]

    def __len__(self) -> int:
        return len(self.windows) - 1

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.windows[index], self.windows[index + 1]

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        for i in range(len(self)):
            yield self[i]
