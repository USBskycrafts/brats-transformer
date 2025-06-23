import unittest

import torch

from transformer.infer import MultiContrastGenerationInferer
from transformer.transformer import ContrastGenerationTransformer
from autoencoder.vqvae import VQGAN


class TestInferer2D(unittest.TestCase):
    def setUp(self):
        # 使用较小的参数加速测试
        # 先创建VQ-VAE来确定实际的latent尺寸
        self.vqvae = VQGAN(
            spatial_dims=2,
            embedding_dim=5
        ).vqvae

        # 测试VQ-VAE的输出尺寸
        test_input = torch.randn(1, 1, 96, 96)
        with torch.no_grad():
            test_feature = self.vqvae.encode_stage_2_inputs(test_input)
        actual_latent_size = test_feature.shape[2:]  # 获取实际的空间尺寸

        self.inferer = MultiContrastGenerationInferer(
            spatial_dims=2,
            latent_dims=5,
            latent_size=actual_latent_size,  # 使用实际尺寸
            hidden_dim=512,
            num_contrasts=4
        )

        self.transformer = ContrastGenerationTransformer(
            num_layers=2,
            num_embeddings=(8 * 8 * 8 * 6 * 5),
            hidden_size=512
        )

    def test_train(self):
        """测试训练forward pass"""
        # 使用较小的图像和batch size
        img = torch.randn(2, 1, 96, 96)
        imgs = [img for _ in range(3)]
        target = torch.randn(2, 1, 96, 96)
        input_contrasts = torch.randint(0, 4, (2, 3))
        target_contrasts = torch.randint(0, 4, (2, 1))

        loss, logits_masked, indices_masked = self.inferer(
            imgs, input_contrasts, target, target_contrasts, self.vqvae, self.transformer)

        # 验证输出形状和值范围
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        self.assertLess(torch.max(indices_masked), 8 * 8 * 8 * 6 * 5)
        self.assertGreaterEqual(torch.min(indices_masked), 0)

        print(f"2D Train - Loss: {loss.item():.4f}")
        print(
            f"2D Train - Logits shape: {logits_masked.shape}, Indices shape: {indices_masked.shape}")

        # 测试反向传播
        loss.backward()

    def test_sample(self):
        """测试采样功能"""
        img = torch.randn(2, 1, 96, 96)
        imgs = [img for _ in range(3)]
        input_contrasts = torch.randint(0, 4, (2, 3))
        target_contrasts = torch.randint(0, 4, (2, 1))

        # 使用较少的迭代次数加速测试
        generated = self.inferer.sample(
            imgs, input_contrasts, target_contrasts, self.vqvae, self.transformer, iterations=4
        )

        # 验证输出形状
        self.assertEqual(generated.shape, (2, 1, 96, 96))
        print(f"2D Sample - Generated shape: {generated.shape}")
        print(
            f"2D Sample - Value range: [{generated.min():.4f}, {generated.max():.4f}]")

    def test_mask_token(self):
        """测试可学习的mask token"""
        mask_token = self.inferer.mask_token

        # 验证mask token不是全零
        self.assertFalse(torch.allclose(
            mask_token, torch.zeros_like(mask_token)))

        # 验证mask token有正确的形状
        self.assertEqual(mask_token.shape, (1, 1, 512))

        # 验证mask token需要梯度
        self.assertTrue(mask_token.requires_grad)

        print(f"2D Mask Token - Shape: {mask_token.shape}")
        print(f"2D Mask Token - Requires grad: {mask_token.requires_grad}")

    def test_different_iterations(self):
        """测试不同迭代次数的采样"""
        img = torch.randn(1, 1, 96, 96)
        imgs = [img for _ in range(3)]
        input_contrasts = torch.randint(0, 4, (1, 3))
        target_contrasts = torch.randint(0, 4, (1, 1))

        for iterations in [1, 2, 4, 8]:
            with self.subTest(iterations=iterations):
                generated = self.inferer.sample(
                    imgs, input_contrasts, target_contrasts, self.vqvae, self.transformer, iterations=iterations
                )
                self.assertEqual(generated.shape, (1, 1, 96, 96))
                print(f"2D Iterations {iterations} - Generated successfully")


class TestInferer3D(unittest.TestCase):
    def setUp(self):
        # 3D配置，先创建VQ-VAE来确定实际的latent尺寸
        self.vqvae = VQGAN(
            spatial_dims=3,
            embedding_dim=5
        ).vqvae

        # 测试VQ-VAE的输出尺寸
        test_input = torch.randn(1, 1, 64, 96, 96)
        with torch.no_grad():
            test_feature = self.vqvae.encode_stage_2_inputs(test_input)
        actual_latent_size = test_feature.shape[2:]  # 获取实际的空间尺寸

        self.inferer = MultiContrastGenerationInferer(
            spatial_dims=3,
            latent_dims=5,
            latent_size=actual_latent_size,  # 使用实际尺寸
            hidden_dim=256,
            num_contrasts=4
        )

        self.transformer = ContrastGenerationTransformer(
            num_layers=2,
            num_embeddings=(8 * 8 * 8 * 6 * 5),
            hidden_size=256
        )

    def test_train(self):
        """测试3D训练forward pass"""
        # 使用小的3D图像
        img = torch.randn(1, 1, 64, 96, 96)  # 减小尺寸和batch size
        imgs = [img for _ in range(3)]
        target = torch.randn(1, 1, 64, 96, 96)
        input_contrasts = torch.randint(0, 4, (1, 3))
        target_contrasts = torch.randint(0, 4, (1, 1))

        loss, logits_masked, indices_masked = self.inferer(
            imgs, input_contrasts, target, target_contrasts, self.vqvae, self.transformer)

        # 验证输出
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        self.assertLess(torch.max(indices_masked), 8 * 8 * 8 * 6 * 5)
        self.assertGreaterEqual(torch.min(indices_masked), 0)

        print(f"3D Train - Loss: {loss.item():.4f}")
        print(
            f"3D Train - Logits shape: {logits_masked.shape}, Indices shape: {indices_masked.shape}")

        # 测试反向传播
        loss.backward()

    def test_sample(self):
        """测试3D采样功能"""
        img = torch.randn(1, 1, 64, 96, 96)
        imgs = [img for _ in range(3)]
        input_contrasts = torch.randint(0, 4, (1, 3))
        target_contrasts = torch.randint(0, 4, (1, 1))

        # 使用较少的迭代次数
        generated = self.inferer.sample(
            imgs, input_contrasts, target_contrasts, self.vqvae, self.transformer, iterations=2
        )

        # 验证输出形状
        self.assertEqual(generated.shape, (1, 1, 64, 96, 96))
        print(f"3D Sample - Generated shape: {generated.shape}")
        print(
            f"3D Sample - Value range: [{generated.min():.4f}, {generated.max():.4f}]")

    def test_mask_token_3d(self):
        """测试3D mask token"""
        mask_token = self.inferer.mask_token

        # 验证mask token不是全零
        self.assertFalse(torch.allclose(
            mask_token, torch.zeros_like(mask_token)))

        # 验证mask token有正确的形状
        self.assertEqual(mask_token.shape, (1, 1, 256))

        # 验证mask token需要梯度
        self.assertTrue(mask_token.requires_grad)

        print(f"3D Mask Token - Shape: {mask_token.shape}")
        print(f"3D Mask Token - Requires grad: {mask_token.requires_grad}")


class TestInfererEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""

    def setUp(self):
        # 先创建VQ-VAE来确定实际的latent尺寸
        self.vqvae = VQGAN(
            spatial_dims=2,
            embedding_dim=5
        ).vqvae

        # 测试VQ-VAE的输出尺寸
        test_input = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            test_feature = self.vqvae.encode_stage_2_inputs(test_input)
        actual_latent_size = test_feature.shape[2:]  # 获取实际的空间尺寸

        self.inferer = MultiContrastGenerationInferer(
            spatial_dims=2,
            latent_dims=5,
            latent_size=actual_latent_size,  # 使用实际尺寸
            hidden_dim=256,
            num_contrasts=4
        )

        self.transformer = ContrastGenerationTransformer(
            num_layers=1,
            num_embeddings=(8 * 8 * 8 * 6 * 5),
            hidden_size=256
        )

    def test_single_batch(self):
        """测试单个batch的情况"""
        img = torch.randn(1, 1, 64, 64)
        imgs = [img for _ in range(3)]
        target = torch.randn(1, 1, 64, 64)
        input_contrasts = torch.randint(0, 4, (1, 3))
        target_contrasts = torch.randint(0, 4, (1, 1))

        # 训练测试
        loss, logits_masked, indices_masked = self.inferer(
            imgs, input_contrasts, target, target_contrasts, self.vqvae, self.transformer)
        self.assertIsInstance(loss, torch.Tensor)

        # 采样测试
        generated = self.inferer.sample(
            imgs, input_contrasts, target_contrasts, self.vqvae, self.transformer, iterations=2
        )
        self.assertEqual(generated.shape, (1, 1, 64, 64))

        print("Edge Case - Single batch test passed")

    def test_consistency(self):
        """测试forward和sample的一致性"""
        img = torch.randn(2, 1, 64, 64)
        imgs = [img for _ in range(3)]
        input_contrasts = torch.randint(0, 4, (2, 3))
        target_contrasts = torch.randint(0, 4, (2, 1))

        # 多次采样应该能正常工作
        for i in range(3):
            with self.subTest(run=i):
                generated = self.inferer.sample(
                    imgs, input_contrasts, target_contrasts, self.vqvae, self.transformer, iterations=2
                )
                self.assertEqual(generated.shape, (2, 1, 64, 64))

        print("Edge Case - Consistency test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
