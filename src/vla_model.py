"""Arquitectura VLA completa: CLIP congelado, fusión Transformer y decoder."""

from typing import Optional

import torch
from torch import nn

from .action_decoder import ActionDecoder
from .transformer import MultimodalTransformer


class VLA(nn.Module):
    """Modelo VLA reutilizable con embeddings precalculados o entradas de CLIP."""

    def __init__(
        self,
        clip_encoder=None,
        embedding_dim: int = 512,
        fusion_dim: int = 256,
        decoder_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.clip_encoder = clip_encoder
        self.fusion = MultimodalTransformer(embedding_dim, fusion_dim)
        self.decoder = ActionDecoder(fusion_dim, decoder_hidden_dim, 8)

        # El encoder se usa solo para generar features; sus pesos no se entrenan.
        if self.clip_encoder is not None:
            for parameter in self.clip_encoder.modelo.parameters():
                parameter.requires_grad = False
            self.clip_encoder.modelo.eval()

    def forward(
        self,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Genera acciones a partir de embeddings o, si se proporcionan, de entradas CLIP."""
        if image_embeddings is None or text_embeddings is None:
            if self.clip_encoder is None or images is None or text_tokens is None:
                raise ValueError("Proporciona ambos embeddings o images, text_tokens y clip_encoder")
            image_embeddings = self.clip_encoder.encode_image(images)
            text_embeddings = self.clip_encoder.encode_text(text_tokens)

        fused = self.fusion(image_embeddings.float(), text_embeddings.float())
        return self.decoder(fused)


VLAModel = VLA
