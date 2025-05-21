import unittest

import torch


from transformer.infer import MultiContrastGenerationInferer
from transformer.transformer import ContrastGenerationTransformer
from autoencoder.vqvae import VQGAN


class TestInferer(unittest.TestCase):
    def setUp(self):
        self.inferer = MultiContrastGenerationInferer()
        self.vqvae = VQGAN(spatial_dims=2,
                           ).vqvae

        self.transformer = ContrastGenerationTransformer(
            in_channels=3,
            img_size=(12, 12),
            spatial_dim=2,
            num_contrast=4,
            num_embeddings=1024
        )

    def test_train(self):
        img = torch.randn(4, 1, 192, 192)
        img = [img for _ in range(3)]
        target = torch.randn(4, 1, 192, 192)
        contrasts = torch.randint(0, 4, (4, 1))
        loss, y, indices = self.inferer(
            img, contrasts, target, self.vqvae, self.transformer)
        self.assertEqual(y.shape, (4, (12 * 12), 1024))
        self.assertLess(indices.max(), 1024)
        self.assertGreaterEqual(indices.min(), 0)
        loss.backward()

    def test_sample(self):
        img = torch.randn(4, 1, 192, 192)
        img = [img for _ in range(3)]
        contrasts = torch.randint(0, 4, (4, 1))
        target = self.inferer.sample(
            img, contrasts, self.vqvae, self.transformer)
        self.assertEqual(target.shape, (4, 1, 192, 192))


class Test3DInferer(unittest.TestCase):
    def setUp(self):
        self.inferer = MultiContrastGenerationInferer()
        self.vqvae = VQGAN(spatial_dims=3,
                           ).vqvae

        self.transformer = ContrastGenerationTransformer(
            in_channels=3,
            img_size=(8, 12, 12),
            spatial_dim=3,
            num_contrast=4,
            num_embeddings=1024
        )

    def test_train(self):
        img = torch.randn(4, 1, 128, 192, 192)
        img = [img for _ in range(3)]
        target = torch.randn(4, 1, 128, 192, 192)
        contrasts = torch.randint(0, 4, (4, 1))
        loss, y, indices = self.inferer(
            img, contrasts, target, self.vqvae, self.transformer)
        self.assertEqual(y.shape, (4, (8 * 12 * 12), 1024))
        self.assertLess(indices.max(), 1024)
        self.assertGreaterEqual(indices.min(), 0)
        loss.backward()

    def test_sample(self):
        img = torch.randn(4, 1, 128, 192, 192)
        img = [img for _ in range(3)]
        contrasts = torch.randint(0, 4, (4, 1))
        target = self.inferer.sample(
            img, contrasts, self.vqvae, self.transformer)
        self.assertEqual(target.shape, (4, 1, 128, 192, 192))
