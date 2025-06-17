import unittest

import torch

from transformer.transformer import ContrastGenerationTransformer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = ContrastGenerationTransformer(
            hidden_size=512,
            mlp_dim=2048,
            num_layers=12,
            num_heads=8,
            num_embeddings=512,
        )

    def test_forward(self):
        # Test the forward pass of the transformer
        x = torch.randn(2, 4 * 12 * 12 + 1, 512)
        y = self.transformer(x)
        self.assertEqual(y.shape, (2, 4 * 12 * 12 + 1, 512))
