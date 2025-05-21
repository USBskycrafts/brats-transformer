from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock


class ContrastGenerationTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: Sequence[int] | int,
                 spatial_dim: int = 3,
                 num_contrast: int = 4,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 ):

        super().__init__()

        self.patch_embed = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=(1, 1) if spatial_dim == 2 else (1, 1, 1),
            hidden_size=hidden_size,
            num_heads=num_heads,
            spatial_dims=spatial_dim,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu',
                norm_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.contrast_embeding = nn.Embedding(
            num_embeddings=num_contrast,
            embedding_dim=hidden_size
        )

        self.out_proj = nn.Linear(hidden_size, num_embeddings)

    def forward(self, imgs: Sequence[torch.Tensor], contrast: torch.Tensor):
        """
        Args:
            imgs: [[B, C, H, W[, D]], ...]
            contrast: [B, 1]
        """

        features = []
        for img in imgs:
            features.append(self.patch_embed(img))
        n = features[-1].shape[1]

        features = torch.cat(features, dim=1)
        contrast = self.contrast_embeding(contrast).repeat(1, n, 1)
        features = torch.cat([features, contrast], dim=1)

        return self.out_proj(self.encoder(features)[:, -n:, :])
