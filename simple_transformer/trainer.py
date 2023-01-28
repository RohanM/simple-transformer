from torch import optim
from torch.utils.data import DataLoader
from simple_transformer.model import Model
from simple_transformer.data import Data
import wandb
from typing import Any

class Trainer:
    model: Model
    optimiser: optim.Optimizer
    wandb: Any
    bs: int

    def __init__(self, model: Model, wandb: Any, lr: float = 1e-3, bs: int = 64) -> None:
        self.model = model
        self.optimiser = optim.AdamW(model.parameters(), lr=lr)
        self.wandb = wandb
        self.bs = bs

    def train(self, num_epochs: int, data: Data) -> None:
        data_loader = DataLoader(data, batch_size=self.bs, shuffle=True)
        self.model.train()
        for epoch in range(num_epochs):
            for x, y in data_loader:
                self.optimiser.zero_grad()
                y_hat, loss = self.model(x, y)
                loss.backward()
                self.optimiser.step()
            self.wandb.log({ "loss": loss.item() })
            print(loss)
