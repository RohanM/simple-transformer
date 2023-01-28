import torch
from torch import optim
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

    def __init__(self, model: TransformerModel, wandb: Any, lr: float = 1e-3, bs: int = 64) -> None:
        self.model = model
        self.optimiser = optim.AdamW(model.parameters(), lr=lr)
        self.wandb = wandb
        self.bs = bs

    def train(self, num_epochs: int, data_train: Data, data_valid: Data) -> None:
        dl_train = DataLoader(data_train, batch_size=self.bs, shuffle=True)
        dl_valid = DataLoader(data_valid, batch_size=self.bs, shuffle=False)

        for epoch in range(num_epochs):
            # -- Train
            self.model.train()
            losses = torch.zeros(len(dl_train))
            for i, (x, y) in enumerate(dl_train):
                y_hat, loss = self.model(x, y)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                losses[i] = loss.item()
            self.wandb.log({ "loss/train": losses.mean() })

            # -- Eval
            self.model.eval()
            losses = torch.zeros(len(dl_valid))
            with torch.no_grad():
                for i, (x, y) in enumerate(dl_valid):
                    y_hat, loss = self.model(x, y)
                    losses[i] = loss.item()
                self.wandb.log({ "loss/valid": losses.mean() })
