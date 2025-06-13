import unittest
from typing import override

import torch
from monai.networks.layers.vector_quantizer import (EMAQuantizer,
                                                    VectorQuantizer)

from autoencoder.quantizer import FiniteScalarQuantizer


class TestQuantizer(unittest.TestCase):

    @override
    def setUp(self):
        self.vq = VectorQuantizer(
            EMAQuantizer(
                spatial_dims=2,
                num_embeddings=1024,
                embedding_dim=4,
            )
        )

        self.fsq = FiniteScalarQuantizer(
            spatial_dims=2,
            levels=[8, 5, 5, 5]
        )

    def test_shape(self):
        x = torch.randn(32, 4, 18, 18)
        vqloss, vq_quantized = self.vq(x)
        fsqloss, fsq_quantized = self.fsq(x)
        self.assertEqual(vq_quantized.shape, fsq_quantized.shape)

        vq_quantized = self.vq.quantize(vq_quantized)
        fsq_quantized = self.fsq.quantize(fsq_quantized)
        self.assertEqual(vq_quantized.shape, fsq_quantized.shape)
        self.assertGreaterEqual(vq_quantized.min(), 0)
        self.assertLessEqual(vq_quantized.max(), 1023)
        self.assertGreaterEqual(fsq_quantized.min(), 0)
        self.assertLessEqual(fsq_quantized.max(), 8 * 5 * 5 * 5 - 1)

        vq_embed = self.vq.embed(vq_quantized)
        fsq_embed = self.fsq.embed(fsq_quantized)
        self.assertEqual(vq_embed.shape, fsq_embed.shape)
