import torch
from torch import tensor
from typing import Optional

from simple_transformer.letter_tokeniser import LetterTokeniser as Tokeniser
from simple_transformer.transformer_model import TransformerModel

class LM:
    tokeniser: Tokeniser
    model: TransformerModel

    NUM_EMBEDDINGS = 32
    CONTEXT_SIZE = 8
    NUM_HEADS = 4
    NUM_BLOCKS = 3

    def __init__(self, filename: Optional[str] = None) -> None:
        self.tokeniser = Tokeniser()
        self.model = TransformerModel(self.tokeniser.vocab_size(), self.CONTEXT_SIZE, self.NUM_EMBEDDINGS, self.NUM_HEADS, self.NUM_BLOCKS, 0)
        if filename is not None:
           self.model.load(filename)
           self.model.eval()
        pass

    def query(self, prompt: str, response_len: int = 100) -> str:
        if prompt == '':
            raise ValueError('Prompt cannot be empty')

        tokens = tensor(self.tokeniser.encode(prompt), dtype=torch.long)
        for _ in range(response_len):
            output = self.model.infer_one(tokens[-self.CONTEXT_SIZE:])
            tokens = torch.cat((tokens, output))
        return self.tokeniser.decode(tokens.tolist())
