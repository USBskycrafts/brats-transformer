import torch.nn as nn
from transformers import RoFormerModel, RoFormerConfig


class ContrastGenerationTransformer(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 num_embeddings: int = 512,
                 ):
        super().__init__()

        config = RoFormerConfig(
            vocab_size=num_embeddings,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            num_hidden_layers=num_layers,
        )
        self.roformer = RoFormerModel(config)

        self.blocks = nn.Sequential(
            # nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(
            #         d_model=hidden_size,
            #         nhead=num_heads,
            #         dim_feedforward=mlp_dim,
            #         dropout=0.1,
            #         activation='gelu',
            #         norm_first=True,
            #         batch_first=True
            #     ),
            #     num_layers=num_layers,
            #     enable_nested_tensor=False
            # ),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings)
        )

    def forward(self, x):
        x = self.roformer(inputs_embeds=x).last_hidden_state
        return self.blocks(x)
