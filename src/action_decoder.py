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
        expected_dim = self.network[0].in_features
        if fused_embedding.ndim != 2 or fused_embedding.shape[-1] != expected_dim:
            raise ValueError(f"La entrada del decodificador debe tener forma (batch, {expected_dim})")
        return self.network(fused_embedding)


ActionMLP = ActionDecoder
