import nltk
from torch import tensor, Tensor
from torch.utils.data import Dataset
from simple_transformer.tokeniser import Tokeniser
from typing import Iterator

class Data(Dataset[tuple[Tensor, Tensor]]):
    windows: list[Tensor]

    def __init__(self, text: str, window_size: int) -> None:
        tokens = Tokeniser().encode(text)
        self.windows = [tensor(ngram) for ngram in nltk.ngrams(tokens, window_size)]

    def __len__(self) -> int:
        return len(self.windows) - 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.windows[index], self.windows[index + 1]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for i in range(len(self)):
            yield self[i]
