import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
from simple_transformer.transformer_model import TransformerModel
from simple_transformer.data import Data
import wandb
from typing import Any

class Trainer:
    model: TransformerModel
    optimiser: optim.Optimizer
    wandb: Any
    bs: int
    eval_interval: int

    def __init__(self, model: TransformerModel, wandb: Any, lr: float = 1e-3, bs: int = 64, eval_interval: int = 100) -> None:
        self.model = model
        self.optimiser = optim.AdamW(model.parameters(), lr=lr)
        self.wandb = wandb
        self.bs = bs
        self.eval_interval = eval_interval

    def train(self, num_epochs: int, data_train: Data, data_valid: Data) -> None:
        dl_train = DataLoader(data_train, batch_size=self.bs, shuffle=True)
        dl_valid = DataLoader(data_valid, batch_size=self.bs, shuffle=False)

        self.model.train()
        for epoch in range(num_epochs):
            losses = torch.zeros(self.eval_interval)
            for i, (x, y) in enumerate(dl_train):
                # -- Log and evaluate
                if i % self.eval_interval == 0 and i > 0:
                    eval_loss = self.eval(dl_valid)
                    self.wandb.log({
                        "loss/train": losses.mean(),
                        "loss/valid": eval_loss,
                    })
                    losses = torch.zeros(self.eval_interval)

                # -- Train
                loss = self.step(x, y)
                losses[i % self.eval_interval] = loss

    def step(self, x: Tensor, y: Tensor) -> Tensor:
        y_hat, loss = self.model(x, y)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss # type: ignore

    @torch.no_grad() # type: ignore
    def eval(self, dl_valid: DataLoader[tuple[Tensor, Tensor]]) -> Tensor:
        self.model.eval()
        losses = torch.zeros(len(dl_valid))
        for i, (x, y) in enumerate(dl_valid):
            y_hat, loss = self.model(x, y)
            losses[i] = loss.item()
        self.model.train()
        return losses.mean()
