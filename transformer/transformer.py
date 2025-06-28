from typing import Optional
import torch
import torch.nn as nn

from transformers import RoFormerConfig, RoFormerModel


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

        config = RoFormerConfig(
            vocab_size=num_embeddings,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            max_position_embeddings=max_seq_len,
            type_vocab_size=1,
            attention_probs_dropout_prob=dropout
        )

        self.transformer_encoder = RoFormerModel(
            config
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

        # Passing through the transformer blocks
        hidden_state = self.transformer_encoder(
            input_ids=indices).last_hidden_state

        # Projecting to the output space
        logits = self.out_proj(hidden_state)
        return logits
