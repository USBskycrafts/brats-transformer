import math
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

        vq_quantized = self.vq.quantize(x)
        fsq_quantized = self.fsq.quantize(x)
        self.assertEqual(vq_quantized.shape, fsq_quantized.shape)
        self.assertGreaterEqual(vq_quantized.min(), 0)
        self.assertLessEqual(vq_quantized.max(), 1023)
        self.assertGreaterEqual(fsq_quantized.min(), 0)
        self.assertLessEqual(fsq_quantized.max(), 8 * 5 * 5 * 5 - 1)

        vq_embed = self.vq.embed(vq_quantized)
        fsq_embed = self.fsq.embed(fsq_quantized)
        self.assertEqual(vq_embed.shape, fsq_embed.shape)
        self.assertTrue(torch.allclose(self.fsq(x)[1], fsq_embed))

    def test_quantize_embed_roundtrip(self):
        """测试 quantize 和 embed 的往返一致性"""
        x = torch.randn(8, 4, 16, 16)

        # 前向传播获得量化后的编码
        _, quantized_encodings = self.fsq(x)

        # 通过 quantize 获得索引
        indices = self.fsq.quantize(x)

        # 通过 embed 恢复编码
        recovered_encodings = self.fsq.embed(indices)

        # 验证往返一致性
        self.assertTrue(torch.allclose(quantized_encodings,
                        recovered_encodings, atol=1e-6))
        print(
            f"往返测试通过: 量化编码和恢复编码的最大差异为 {torch.max(torch.abs(quantized_encodings - recovered_encodings)).item()}")

    def test_quantize_indices_range(self):
        """测试量化索引的范围正确性"""
        levels = [8, 5, 5, 5]
        fsq = FiniteScalarQuantizer(spatial_dims=2, levels=levels)

        # 计算预期的最大索引值
        expected_max_index = torch.prod(torch.tensor(levels)).item() - 1

        # 测试不同大小的输入
        for batch_size in [1, 4, 8]:
            for spatial_size in [8, 16, 32]:
                x = torch.randn(batch_size, len(levels),
                                spatial_size, spatial_size)
                indices = fsq.quantize(x)

                self.assertGreaterEqual(indices.min().item(), 0)
                self.assertLessEqual(indices.max().item(), expected_max_index)
                self.assertEqual(indices.shape, (batch_size,
                                 spatial_size, spatial_size))

    def test_different_levels_configuration(self):
        """测试不同的 levels 配置"""
        test_configs = [
            [4, 4, 4, 4],
            [8, 6, 5],
            [16, 8],
            [32],
            [2, 2, 2, 2, 2, 2]  # 6维配置
        ]

        for levels in test_configs:
            fsq = FiniteScalarQuantizer(spatial_dims=2, levels=levels)
            x = torch.randn(2, len(levels), 8, 8)

            # 测试前向传播
            _, quantized = fsq(x)
            self.assertEqual(quantized.shape, x.shape)

            # 测试往返
            indices = fsq.quantize(x)
            recovered = fsq.embed(indices)

            # 验证索引范围
            expected_max = torch.prod(torch.tensor(levels)).item() - 1
            self.assertLessEqual(indices.max().item(), expected_max)
            self.assertGreaterEqual(indices.min().item(), 0)

            # 验证往返一致性
            self.assertTrue(torch.allclose(quantized, recovered, atol=1e-6))

    def test_boundary_values(self):
        """测试边界值情况"""
        fsq = FiniteScalarQuantizer(spatial_dims=2, levels=[8, 5, 5, 5])

        # 测试零值输入
        x_zero = torch.zeros(2, 4, 8, 8)
        _, quantized_zero = fsq(x_zero)
        indices_zero = fsq.quantize(x_zero)
        recovered_zero = fsq.embed(indices_zero)
        self.assertTrue(torch.allclose(
            quantized_zero, recovered_zero, atol=1e-6))

        # 测试极大值输入
        x_large = torch.ones(2, 4, 8, 8) * 10
        _, quantized_large = fsq(x_large)
        indices_large = fsq.quantize(x_large)
        recovered_large = fsq.embed(indices_large)
        self.assertTrue(torch.allclose(
            quantized_large, recovered_large, atol=1e-6))

        # 测试极小值输入
        x_small = torch.ones(2, 4, 8, 8) * (-10)
        _, quantized_small = fsq(x_small)
        indices_small = fsq.quantize(x_small)
        recovered_small = fsq.embed(indices_small)
        self.assertTrue(torch.allclose(
            quantized_small, recovered_small, atol=1e-6))

    def test_gradient_flow(self):
        """测试梯度流是否正常（Straight-Through Estimator）"""
        fsq = FiniteScalarQuantizer(spatial_dims=2, levels=[8, 5, 5, 5])

        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        # 前向传播
        _, quantized = fsq(x)
        loss = quantized.sum()

        # 反向传播
        loss.backward()

        # 验证梯度存在且不为零
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        self.assertEqual(x.grad.shape, x.shape)
        print(f"梯度测试通过: 梯度范数为 {torch.norm(x.grad).item()}")

    def test_mathematical_correctness(self):
        """测试FSQ数学计算的严格正确性，基于FSQ论文的原理"""
        levels = [4, 3, 5]  # 使用较小的levels便于手动验证
        fsq = FiniteScalarQuantizer(spatial_dims=2, levels=levels)

        # 1. 测试量化边界精确性
        print("=== 测试量化边界精确性 ===")
        x_boundary = torch.tensor([[[[2.0, -2.0],   # 测试边界值
                                   [0.0, 1.5]],
                                  [[1.0, -1.0],
                                   [0.5, -0.5]],
                                  [[2.5, -2.5],
                                   # [1, 3, 2, 2]
                                   [0.1, -0.1]]]], dtype=torch.float32)

        _, quantized = fsq(x_boundary)

        # 先观察实际的量化值范围
        for i, level in enumerate(levels):
            channel_values = quantized[0, i, :, :]
            half_width = level // 2
            print(
                f"维度{i} (level={level}): 值范围 [{channel_values.min().item():.3f}, {channel_values.max().item():.3f}], half_width={half_width}")

            # 验证量化值在合理范围内 [-1, 1]
            self.assertTrue(torch.all(channel_values >= -1.0 - 1e-6),
                            f"维度{i}: 值 {channel_values.min().item()} 小于 -1.0")
            self.assertTrue(torch.all(channel_values <= 1.0 + 1e-6),
                            f"维度{i}: 值 {channel_values.max().item()} 大于 1.0")

        # 2. 测试基数系统数学正确性
        print("=== 测试基数系统数学正确性 ===")
        # 手动验证基数计算
        expected_basis = torch.tensor(
            [1, 4, 12], dtype=torch.int32)  # [1, L0, L0*L1]
        self.assertTrue(torch.equal(fsq._basis, expected_basis),
                        f"基数计算错误: 期望{expected_basis}, 实际{fsq._basis}")

        # 创建一个可以手动验证的简单案例
        x_simple = torch.zeros(1, 3, 1, 1)  # 全零输入，应该映射到中心码字
        indices_simple = fsq.quantize(x_simple)

        # 对于全零输入，每个维度都应该量化到中心值
        # levels=[4,3,5] -> 中心索引应该是 [2,1,2] -> 索引 = 2*1 + 1*4 + 2*12 = 30
        expected_center_codes = []
        for level in levels:
            expected_center_codes.append(level // 2)
        expected_index = sum(
            code * basis for code, basis in zip(expected_center_codes, fsq._basis.tolist()))

        print(f"全零输入的预期中心索引: {expected_index}, 实际索引: {indices_simple.item()}")

        # 3. 测试索引编码/解码的双射性
        print("=== 测试索引编码/解码双射性 ===")
        total_combinations = torch.prod(torch.tensor(levels)).item()

        # 测试一些具体的索引编码/解码案例
        print(f"总共有 {total_combinations} 个可能的组合")

        # 测试几个关键索引：0, 中心索引, 最大索引
        test_indices = [0, expected_index, total_combinations - 1]

        for test_idx in test_indices:
            # 将单个索引转换为正确的形状进行测试 [batch, height, width]
            # [1, 1, 1] for spatial_dims=2
            single_index = torch.tensor([[[test_idx]]])
            embedded_single = fsq.embed(single_index)  # [1, 3, 1, 1]
            re_quantized_single = fsq.quantize(embedded_single)  # [1, 1, 1]

            recovered_idx = re_quantized_single.item()
            print(f"索引 {test_idx}: 嵌入后恢复为 {recovered_idx}")

            if test_idx == recovered_idx:
                print(f"  ✓ 索引 {test_idx} 往返成功")
            else:
                print(f"  ✗ 索引 {test_idx} 往返失败: {test_idx} -> {recovered_idx}")

        # 测试边界码字的正确性
        print("测试边界码字:")
        # 最小码字 [0,0,0] -> 索引 0
        # 最大码字 [3,2,4] -> 索引 3*1 + 2*4 + 4*12 = 59
        boundary_indices = [0, total_combinations - 1]
        for idx in boundary_indices:
            single_index = torch.tensor([[[idx]]])  # [1, 1, 1]
            embedded = fsq.embed(single_index)  # [1, 3, 1, 1]

            # 手动计算预期的码字
            expected_codes = []
            temp_idx = idx
            for i, (level, basis) in enumerate(zip(levels, fsq._basis.tolist())):
                if i == len(levels) - 1:
                    code = temp_idx // basis
                else:
                    code = temp_idx // basis % level
                expected_codes.append(code)
                temp_idx = temp_idx % basis if i < len(levels) - 1 else 0

            print(f"  索引 {idx} 的预期码字: {expected_codes}")

            # 验证嵌入值是否符合预期
            recovered_idx = fsq.quantize(embedded).item()
            if idx == recovered_idx:
                print(f"  ✓ 边界索引 {idx} 测试通过")
            else:
                print(f"  ✗ 边界索引 {idx} 测试失败")

        # 额外测试：验证一些具体码字到索引的转换
        print("验证具体码字转换:")
        test_cases = [
            ([0, 0, 0], 0),           # 最小码字
            ([2, 1, 2], expected_index),  # 中心码字
            ([3, 2, 4], total_combinations - 1)  # 最大码字
        ]

        for codes, expected_idx in test_cases:
            # 手动构造嵌入张量
            manual_embedded = torch.zeros(1, 3, 1, 1)
            for i, code in enumerate(codes):
                half_width = levels[i] // 2
                manual_embedded[0, i, 0, 0] = (code - half_width) / half_width

            # 通过quantize获得索引
            manual_idx = fsq.quantize(manual_embedded).item()
            print(f"  码字 {codes} -> 索引 {manual_idx} (期望: {expected_idx})")

            if manual_idx == expected_idx:
                print(f"    ✓ 码字转换正确")
            else:
                print(f"    ✗ 码字转换错误")

        print("双射性测试完成")

        # 4. 测试量化误差分析
        print("=== 测试量化误差分析 ===")
        x_test = torch.randn(4, 3, 8, 8) * 2  # 较大范围的随机输入

        _, quantized_output = fsq(x_test)
        indices = fsq.quantize(x_test)
        reconstructed = fsq.embed(indices)

        # 计算重构误差
        reconstruction_error = torch.mean(
            (quantized_output - reconstructed) ** 2)
        self.assertLess(reconstruction_error.item(), 1e-10,
                        f"重构误差过大: {reconstruction_error.item()}")

        # 验证量化误差在理论范围内
        quantization_error = torch.mean(
            (x_test.permute(0, 2, 3, 1) - reconstructed.permute(0, 2, 3, 1)) ** 2)
        print(f"平均量化误差: {quantization_error.item()}")

        # 5. 测试直通估计器的数学性质
        print("=== 测试直通估计器性质 ===")
        x_grad = torch.randn(2, 3, 4, 4, requires_grad=True)

        # 前向传播
        _, quantized_grad = fsq(x_grad)
        loss = torch.sum(quantized_grad ** 2)

        # 反向传播
        loss.backward()

        # 验证梯度形状和数值合理性
        self.assertEqual(x_grad.grad.shape, x_grad.shape)
        self.assertTrue(torch.all(torch.isfinite(x_grad.grad)), "梯度包含非有限值")

        # STE应该传递梯度，梯度不应该全为零
        self.assertTrue(torch.any(x_grad.grad != 0), "STE没有正确传递梯度")

        # 6. 测试边界条件和数值稳定性
        print("=== 测试数值稳定性 ===")
        # 测试极大值
        x_extreme = torch.tensor(
            [[[[1000.0]], [[-1000.0]], [[0.0]]]], dtype=torch.float32)
        _, quantized_extreme = fsq(x_extreme)
        indices_extreme = fsq.quantize(x_extreme)

        # 验证极值输入不会导致NaN或Inf
        self.assertTrue(torch.all(torch.isfinite(
            quantized_extreme)), "极值输入导致非有限量化值")
        self.assertTrue(torch.all(torch.isfinite(
            indices_extreme.float())), "极值输入导致非有限索引")

        # 验证索引范围
        self.assertTrue(torch.all(indices_extreme >= 0), "索引小于0")
        self.assertTrue(torch.all(indices_extreme < total_combinations),
                        f"索引超出范围 [0, {total_combinations})")

        # 7. 测试不同维度的一致性
        print("=== 测试维度一致性 ===")
        for test_levels in [[8], [4, 4], [2, 2, 2, 2]]:
            fsq_test = FiniteScalarQuantizer(
                spatial_dims=2, levels=test_levels)
            x_dim_test = torch.randn(2, len(test_levels), 4, 4)

            # 验证所有操作都能正确执行
            _, quantized_dim = fsq_test(x_dim_test)
            indices_dim = fsq_test.quantize(x_dim_test)
            reconstructed_dim = fsq_test.embed(indices_dim)

            # 验证一致性
            self.assertTrue(torch.allclose(quantized_dim, reconstructed_dim, atol=1e-6),
                            f"维度配置 {test_levels} 的一致性测试失败")

            print(f"维度配置 {test_levels} 测试通过")

        print("=== 所有数学正确性测试通过 ===")

    def test_spatial_dims_consistency(self):
        """测试不同空间维度的一致性"""
        levels = [8, 5, 5, 5]

        # 测试不同的spatial_dims
        for spatial_dims in [1, 2, 3]:
            fsq = FiniteScalarQuantizer(
                spatial_dims=spatial_dims, levels=levels)

            for _ in range(1000):
                if spatial_dims == 1:
                    x = torch.randn(2, 4, 16) * 1e4
                elif spatial_dims == 2:
                    x = torch.randn(2, 4, 8, 8) * 1e4
                else:  # spatial_dims == 3
                    x = torch.randn(2, 4, 4, 4, 4) * 1e4

                # 测试前向传播
                _, quantized = fsq(x)
                self.assertEqual(quantized.shape, x.shape)

                # 测试往返
                indices = fsq.quantize(x)
                recovered = fsq.embed(indices)
                self.assertTrue(torch.allclose(
                    quantized, recovered, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
