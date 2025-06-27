from typing import Optional
import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings)
        )

    def forward(self, indices: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            indices: (B, L) tensor of indices to be transformed.
        Returns:
            logits: (B, L, num_embeddings) tensor of logits for each index.
        """
        # Embedding the input indices
        x = self.embedding(indices)

        # Adding positional embeddings
        seq_len = x.size(1)
        if seq_len > self.position_embedding.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max sequence length {self.position_embedding.size(1)}."
            )
        x += self.position_embedding[:, :seq_len, :]

        # Passing through the transformer blocks
        hidden_state = self.transformer_encoder(x, mask=mask)

        # Projecting to the output space
        logits = self.out_proj(hidden_state)
        return logits
