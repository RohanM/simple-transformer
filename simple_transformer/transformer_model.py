import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import cast, Optional, Tuple

class Head(nn.Module):
    key: nn.Linear
    query: nn.Linear
    value: nn.Linear
    mask: Tensor

    def __init__(self, context_size: int, num_embeddings: int, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape

        k = self.key(x) # (b, t, c)
        q = self.query(x) # (b, t, c)
        v = self.value(x) # (b, t, c)

        weights = q @ k.transpose(-2, -1) * c ** -0.5 # (b, t, c) @ (b, c, t) -> (b, t, t)
        weights = weights.masked_fill(self.mask[:t, :t] == 0, float('-inf')) # (b, t, t)
        weights = F.softmax(weights, dim=-1) # (b, t, t)
        y = weights @ v # (b, t, t) @ (b, t, c) -> (b, t, c)
        return cast(Tensor, y)


class MultiHeadAttention(nn.Module):
    heads: nn.ModuleList
    proj: nn.Linear

    def __init__(self, num_heads: int, context_size: int, num_embeddings: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(context_size, num_embeddings, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_embeddings, num_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    net: nn.Sequential

    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        num_hidden = num_embeddings * 4
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_embeddings),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) # type: ignore


class Block(nn.Module):
    attn_heads: MultiHeadAttention
    feed_forward: FeedForward

    def __init__(self, num_heads: int, context_size: int, num_embeddings: int) -> None:
        super().__init__()
        self.attn_heads = MultiHeadAttention(num_heads, context_size, num_embeddings, num_embeddings // num_heads)
        self.feed_forward = FeedForward(num_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn_heads(x) + x
        x = self.feed_forward(x) + x
        return x


class TransformerModel(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    blocks: nn.ModuleList
    lm_head: nn.Linear

    def __init__(self, vocab_size: int, context_size: int, num_embeddings: int, num_heads: int, num_blocks: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embeddings)
        self.pos_embedding = nn.Embedding(context_size, num_embeddings)
        self.blocks = nn.ModuleList([Block(num_heads, context_size, num_embeddings) for _ in range(num_blocks)])
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        b, t = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(t, device=x.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), target.view(b*t))

        return logits, loss

    def infer_one(self, x: Tensor) -> Tensor:
        # Do not accept batch dimension
        assert x.dim() == 1

        logits, _ = self(x.unsqueeze(0))
        probs = F.softmax(logits[0], dim=-1)
        return torch.multinomial(probs, 1)[-1]

    def save(self, filename: str = 'model.pt') -> None:
        state = {
            'token_embedding': self.token_embedding.state_dict(),
            'pos_embedding': self.pos_embedding.state_dict(),
            'blocks': self.blocks.state_dict(),
            'lm_head': self.lm_head.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename: str = 'model.pt') -> None:
        state = torch.load(filename) # type: ignore
        self.token_embedding.load_state_dict(state['token_embedding'])
        self.pos_embedding.load_state_dict(state['pos_embedding'])
        self.blocks.load_state_dict(state['blocks'])
        self.lm_head.load_state_dict(state['lm_head'])
