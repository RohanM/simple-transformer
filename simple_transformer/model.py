import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import cast, Optional, Tuple

class Model(nn.Module):
    embedding: nn.Embedding

    def __init__(self, num_embeddings: int) -> None:
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, num_embeddings)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        logits = self.embedding(x)

        loss = None
        if target is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), target.view(b*t))

        return logits, loss

    def infer_one(self, x: Tensor) -> Tensor:
        logits, _ = self(x)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)[-1]

    def save(self, filename: str = 'model.pt') -> None:
        state = {
            'embedding': self.embedding.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename: str = 'model.pt') -> None:
        state = torch.load(filename) # type: ignore
        self.embedding.load_state_dict(state['embedding'])
