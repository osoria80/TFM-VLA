"""Transformer de fusion entre los embeddings visual y textual de CLIP."""

import torch
from torch import nn


class MultimodalTransformer(nn.Module):
    """Fusiona dos embeddings de CLIP y devuelve un vector de 256 elementos."""

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim debe ser divisible por num_heads")

        self.image_projection = nn.Linear(embedding_dim, hidden_dim)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim)
        # Hay dos tokens: uno visual y otro textual.
        self.position_embeddings = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.normal_(self.position_embeddings, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Devuelve la media de los dos tokens tras la atención multimodal."""
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("Los embeddings deben tener forma (batch, embedding_dim)")
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError("Los embeddings visual y textual deben tener la misma forma")
        if image_embeddings.shape[-1] != self.image_projection.in_features:
            raise ValueError(f"Se esperaban embeddings de {self.image_projection.in_features} dimensiones")

        tokens = torch.stack(
            [self.image_projection(image_embeddings), self.text_projection(text_embeddings)],
            dim=1,
        )
        fused_tokens = self.encoder(tokens + self.position_embeddings)
        return self.output_norm(fused_tokens.mean(dim=1))


# Alias corto y explícito para reutilizar el módulo en otros notebooks.
FusionTransformer = MultimodalTransformer
