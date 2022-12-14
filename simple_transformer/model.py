import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Model(nn.Module):
    embedding: nn.Embedding
    transformer: nn.Transformer

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.transformer(x, x)
        return x

    def loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.mse_loss(y_hat, y)

    def infer_one(self, x: Tensor) -> Tensor:
        return self.reverse_embedding(self(x).unsqueeze(0)).squeeze()[-1:]

    def embed(self, x: Tensor) -> Tensor:
        return self.embedding(x)

    def reverse_embedding(self, x: Tensor) -> Tensor:
        distances = torch.linalg.norm(self.embedding.weight - x.unsqueeze(2), dim=3)
        idxs = torch.argmin(distances, dim=2)
        return idxs

    def save(self, filename: str = 'model.pt') -> None:
        state = {
            'embedding': self.embedding.state_dict(),
            'transformer': self.transformer.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename: str = 'model.pt') -> None:
        state = torch.load(filename)
        self.embedding.load_state_dict(state['embedding'])
        self.transformer.load_state_dict(state['transformer'])
