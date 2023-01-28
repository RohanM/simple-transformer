import torch
from torch import tensor
from typing import Optional

from simple_transformer.letter_tokeniser import LetterTokeniser as Tokeniser
from simple_transformer.model import Model

class LM:
    tokeniser: Tokeniser
    model: Model

    NUM_EMBEDDINGS = 256

    def __init__(self, filename: Optional[str] = None) -> None:
        self.tokeniser = Tokeniser()
        self.model = Model(self.tokeniser.vocab_size())
        if filename is not None:
           self.model.load(filename)
           self.model.eval()
        pass

    def query(self, prompt: str, response_len: int = 100) -> str:
        tokens = tensor(self.tokeniser.encode(prompt), dtype=torch.long)
        for _ in range(response_len):
           output = self.model.infer_one(tokens)
           tokens = torch.cat((tokens, output))
        return self.tokeniser.decode(tokens.tolist())
