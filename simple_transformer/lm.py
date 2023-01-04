import torch
from torch import LongTensor

from simple_transformer.tokeniser import Tokeniser
from simple_transformer.model import Model

class LM:
    tokeniser: Tokeniser
    model: Model

    NUM_EMBEDDINGS = 256

    def __init__(self, filename: str = None) -> None:
        self.tokeniser = Tokeniser()
        self.model = Model(self.tokeniser.vocab_size(), self.NUM_EMBEDDINGS)
        if filename is not None:
            self.model.load(filename)
            self.model.eval()

    def query(self, prompt: str) -> str:
        tokens = self.tokeniser.encode(prompt)
        tokens_t = LongTensor(tokens).unsqueeze(0)
        output_t = self.model(tokens_t)
        output_t = self.model.reverse_embedding(output_t)
        output = output_t.squeeze().tolist()
        return self.tokeniser.decode(output)
