from torch import nn, Tensor

class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
