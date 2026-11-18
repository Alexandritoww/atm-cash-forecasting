import torch
from torch import nn


class MultiTaskATMNetwork(nn.Module):
    """
    Red neuronal multitarea para predecir la cantidad
    de billetes demandados por denominación.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim debe ser positivo.")

        # Representación compartida.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Una cabeza independiente por denominación.
        self.heads = nn.ModuleList([
            nn.Linear(32, 1)
            for _ in range(4)
        ])

    def forward(self, x):
        shared_features = self.shared(x)

        predictions = [
            head(shared_features)
            for head in self.heads
        ]

        return torch.cat(predictions, dim=1)