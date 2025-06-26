import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 ):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    activation='gelu',
                    norm_first=True,
                    batch_first=True
                ),
                num_layers=num_layers,
                enable_nested_tensor=False
            ),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, num_embeddings)
        )

    def forward(self, x):
        return self.blocks(x)


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 ):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    activation='gelu',
                    norm_first=True,
                    batch_first=True
                ),
                num_layers=num_layers,
                enable_nested_tensor=False
            ),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, num_embeddings)
        )

    def forward(self, x):
        last_hidden_state = self.blocks(x)
        logits = self.out_proj(last_hidden_state)
        return last_hidden_state, logits
