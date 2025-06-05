from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from einops import rearrange


class ContrastGenerationTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: Sequence[int] | int,
                 spatial_dims: int = 3,
                 num_contrast: int = 4,
                 patch_size:  int = 2,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 ):

        super().__init__()
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims

        self.patch_embed = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=(patch_size, ) * spatial_dims,
            hidden_size=hidden_size,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            proj_type='perceptron',
            pos_embed_type='sincos',
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu',
                norm_first=True,
                batch_first=True
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.contrast_embed = nn.Parameter(
            torch.randn(num_contrast, hidden_size),
            requires_grad=True
        )

        self.generated_embed = nn.Parameter(
            torch.randn(torch.prod(torch.tensor(img_size)), hidden_size),
            requires_grad=True
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings * (patch_size ** spatial_dims)) 
        )

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
        contrast = self.contrast_embed[contrast]
        generated_embedding = self.generated_embed.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )

        features = self.encoder(
            torch.cat([features, contrast, generated_embedding], dim=1)
        )
        output_tokens =  self.out_proj(features[:, -n:, :])
        if self.spatial_dims == 2:
            output_shape = f'b h (c ph pw) -> b (h ph pw) c' 
            output_images = rearrange(
                output_tokens,
                output_shape,
                ph=self.patch_size,
                pw=self.patch_size
            )
        elif self.spatial_dims == 3:
            output_shape = f'b h (c ph pw pd) -> b (h ph pw pd) c'
            output_images = rearrange(
                output_tokens,
                output_shape,
                ph=self.patch_size,
                pw=self.patch_size,
                pd=self.patch_size
            )
        else:
            raise ValueError(f"Unsupported spatial dimensions: {self.spatial_dims}")
        return output_images
