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

    def query(self, prompt: str, response_len: int = 100) -> str:
        tokens = self.tokeniser.encode(prompt)
        tokens_t = LongTensor(tokens)
        for _ in range(response_len):
            output = self.model.infer_one(tokens_t)
            tokens = tokens[1:] + [output.item()]
            tokens_t = LongTensor(tokens)
        return self.tokeniser.decode(tokens)

    def query_one_token(self, prompt: str) -> str:
        tokens = self.tokeniser.encode(prompt)
        tokens_t = LongTensor(tokens)
        output = self.model.infer_one(tokens_t)
        return self.tokeniser.decode(output.tolist())
