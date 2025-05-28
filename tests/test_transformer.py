import unittest

import torch

from transformer.transformer import ContrastGenerationTransformer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = ContrastGenerationTransformer(
            in_channels=4,
            img_size=(12, 12),
            num_contrast=4,
            num_embeddings=128,
            spatial_dims=2
        )

    def test_forward(self):
        # Test the forward pass of the transformer
        x = torch.randn(8, 4, 12, 12)
        imgs = [x for _ in range(3)]
        contrasts = torch.randint(0, 4, (8, 1))
        y = self.transformer(imgs, contrasts)
        self.assertEqual(y.shape, (8, (12 * 12), 128))


class Test3DTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = ContrastGenerationTransformer(
            in_channels=4,
            img_size=(12, 12, 12),
            num_contrast=4,
            num_embeddings=128,
            spatial_dims=3
        )

    def test_forward(self):
        # Test the forward pass of the transformer
        x = torch.randn(8, 4, 12, 12, 12)
        imgs = [x for _ in range(3)]
        contrasts = torch.randint(0, 4, (8, 1))
        y = self.transformer(imgs, contrasts)
        self.assertEqual(y.shape, (8, (12 * 12 * 12), 128))
