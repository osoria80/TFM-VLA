"""Decodificador MLP para acciones continuas normalizadas."""

import torch
from torch import nn


class ActionDecoder(nn.Module):
    """Convierte la representación multimodal en ocho componentes [0, 1]."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, output_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        if fused_embedding.ndim != 2 or fused_embedding.shape[-1] != 256:
            raise ValueError("La entrada del decodificador debe tener forma (batch, 256)")
        return self.network(fused_embedding)


ActionMLP = ActionDecoder
