import torch
from torch import LongTensor

from simple_transformer.tokeniser import Tokeniser
from simple_transformer.model import Model

class LM:
    NUM_EMBEDDINGS = 512

    def __init__(self) -> None:
        self.tokeniser = Tokeniser()
        self.model = Model(self.tokeniser.vocab_size(), self.NUM_EMBEDDINGS)

    def query(self, prompt: str) -> str:
        tokens = self.tokeniser.encode(prompt)
        tokens_t = LongTensor(tokens).unsqueeze(0)
        output_t = self.model(tokens_t)
        output = output_t.squeeze().tolist()
        return self.tokeniser.decode(output)
