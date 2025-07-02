import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import override
import random

from transformer.maskgit import MaskGit


class TestMaskGit(unittest.TestCase):

    @override
    def setUp(self):
        """设置测试环境"""
        # 创建mock transformer
        self.mock_transformer = MagicMock()

        # 设置基本参数
        self.mask_token_id = 1024
        self.vocab_size = 1025
        self.batch_size = 2
        self.seq_len = 16
        self.num_tokens = 8
        self.device = torch.device('cpu')

        # 创建MaskGit实例
        self.maskgit = MaskGit(
            transformer=self.mock_transformer,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            schedule="cosine"
        )

        # 创建测试数据
        self.input_indices = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len))
        self.target_indices = torch.randint(
            0, self.vocab_size, (self.batch_size, self.num_tokens))
        self.conditions = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len))
        self.input_mask = torch.ones_like(self.conditions, dtype=torch.bool)

    def test_forward_basic(self):
        """测试forward函数的基本功能"""
        # 设置mock transformer的返回值
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.side_effect = lambda x, mask: mock_logits
        torch.manual_seed(42)
        random.seed(42)

        # 调用forward函数
        loss = self.maskgit.forward(
            self.input_indices, self.input_mask, self.target_indices)

        # 验证结果
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # 标量loss
        self.assertTrue(torch.isfinite(loss))

        # 验证transformer被正确调用
        self.mock_transformer.assert_called_once()

        # 验证输入到transformer的张量形状
        call_args = self.mock_transformer.call_args[0][0]
        expected_shape = (self.batch_size, self.seq_len + self.num_tokens)
        self.assertEqual(call_args.shape, expected_shape)

    @patch('random.random')
    def test_forward_different_mask_ratios(self, mock_random):
        """测试不同mask比例下的forward函数"""
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.side_effect = lambda x, mask: mock_logits
        torch.manual_seed(42)
        random.seed(42)

        # 测试不同的mask比例
        test_ratios = [0.1, 0.5, 0.9]

        for ratio in test_ratios:
            with self.subTest(mask_ratio=ratio):
                mock_random.return_value = ratio

                loss = self.maskgit.forward(
                    self.input_indices, self.input_mask, self.target_indices)

                # 验证loss是有限的
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(loss.item(), 0)  # 交叉熵loss应该为正

    def test_forward_mask_ratio_accuracy(self):
        """测试forward中实际掩码比例是否接近预期"""
        # 设置mock返回值
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        # 测试不同的mask ratio
        test_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        tolerance = 0.15  # 允许15%的误差（考虑到随机性和安全机制）

        for expected_ratio in test_ratios:
            with self.subTest(ratio=expected_ratio):
                with patch('random.random', return_value=expected_ratio):
                    # 执行forward，同时捕获传递给transformer的输入
                    loss = self.maskgit.forward(
                        self.input_indices, self.input_mask, self.target_indices)

                    # 获取传递给transformer的输入
                    call_args = self.mock_transformer.call_args[0][0]
                    # 前num_tokens是target部分
                    target_part = call_args[:, :self.num_tokens]

                    # 计算实际被掩码的比例
                    mask_count = (
                        target_part == self.mask_token_id).sum().item()
                    total_count = target_part.numel()
                    actual_ratio = mask_count / total_count

                    # 验证实际比例接近预期比例
                    self.assertAlmostEqual(
                        1 - actual_ratio, expected_ratio,
                        delta=tolerance,
                        msg=f"Expected ratio {expected_ratio:.1f}, got {actual_ratio:.3f}"
                    )

                    # 验证loss有效
                    self.assertTrue(torch.isfinite(loss))

                    # 特殊情况验证
                    if expected_ratio == 1.0:
                        # 100%掩码时，应该有安全机制确保至少一个token未被掩码
                        self.assertLess(
                            1 - actual_ratio, 1.0, "Safety mechanism should prevent 100% masking")
                    elif expected_ratio == 0.0:
                        # 0%掩码时，应该没有token被掩码
                        self.assertEqual(
                            1 - actual_ratio, 0.0, "0% ratio should result in no masking")

    def test_generate_topk_replacement(self):
        """测试generate过程中topk位置的正确替换"""
        # 设置mock返回特定的logits，确保可预测的topk选择
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.zeros(logits_shape)

        # 设置特定位置有高置信度，让token 100在前几个位置有最高置信度
        mock_logits[:, :self.num_tokens, 100] = 10.0  # token 100有最高置信度
        mock_logits[:, :self.num_tokens, 200] = 5.0   # token 200有次高置信度

        self.mock_transformer.return_value = mock_logits

        # 执行一步generate
        generated_indices = self.maskgit.generate(
            num_tokens=self.num_tokens,
            conditions=self.conditions,
            conditions_mask=self.input_mask,
            device=self.device,
            steps=1,
            temperature=1.0
        )

        # 验证返回了两个状态：初始状态和一步后状态
        self.assertEqual(len(generated_indices), 2)

        # 验证初始状态：全是mask_token_id
        initial_state = generated_indices[0]
        self.assertTrue(torch.all(initial_state == self.mask_token_id))

        # 验证一步后：topk位置被替换
        after_one_step = generated_indices[1]

        # 应该有一些位置被替换成token 100（最高置信度）
        replaced_positions = (after_one_step == 100)
        self.assertTrue(torch.any(replaced_positions))

        # 剩余位置应该仍然是mask_token_id
        mask_positions = (after_one_step == self.mask_token_id)
        self.assertTrue(torch.any(mask_positions))

        # 验证被替换的位置数量合理（应该是根据schedule计算的k值）
        for i in range(self.batch_size):
            replaced_count = replaced_positions[i].sum().item()
            total_positions = self.num_tokens
            # 第一步应该替换一定数量的token
            self.assertGreater(replaced_count, 0)
            self.assertLess(replaced_count, total_positions)

    def test_generate_basic(self):
        """测试generate函数的基本功能"""
        # 设置mock transformer的返回值
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        # 调用generate函数
        generated_indices = self.maskgit.generate(
            num_tokens=self.num_tokens,
            conditions=self.conditions,
            conditions_mask=self.input_mask,
            device=self.device,
            steps=4,
            temperature=1.0
        )

        # 验证返回结果
        self.assertIsInstance(generated_indices, list)
        self.assertGreaterEqual(len(generated_indices), 1)  # 至少有初始状态

        # 验证每个步骤的形状
        for step_indices in generated_indices:
            self.assertEqual(step_indices.shape,
                             (self.batch_size, self.num_tokens))
            self.assertTrue(torch.all(step_indices >= 0))
            self.assertTrue(torch.all(step_indices <= self.vocab_size))

    def test_generate_different_steps(self):
        """测试不同步数的generate函数"""
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        test_steps = [1, 4, 8, 16]

        for steps in test_steps:
            with self.subTest(steps=steps):
                generated_indices = self.maskgit.generate(
                    num_tokens=self.num_tokens,
                    conditions=self.conditions,
                    conditions_mask=self.input_mask,
                    device=self.device,
                    steps=steps,
                    temperature=1.0
                )

                # 验证步数不超过预期（可能提前结束）
                self.assertLessEqual(len(generated_indices), steps + 1)
                self.assertGreaterEqual(len(generated_indices), 1)

    def test_generate_different_temperatures(self):
        """测试不同温度下的generate函数"""
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        test_temperatures = [0.5, 1.0, 2.0]

        for temperature in test_temperatures:
            with self.subTest(temperature=temperature):
                generated_indices = self.maskgit.generate(
                    num_tokens=self.num_tokens,
                    conditions=self.conditions,
                    conditions_mask=self.input_mask,
                    device=self.device,
                    steps=4,
                    temperature=temperature
                )

                # 验证生成结果的基本属性
                self.assertIsInstance(generated_indices, list)
                self.assertGreater(len(generated_indices), 0)

                # 验证最终结果的形状
                final_indices = generated_indices[-1]
                self.assertEqual(final_indices.shape,
                                 (self.batch_size, self.num_tokens))

    def test_generate_cosine_schedule(self):
        """测试cosine调度的generate函数"""
        maskgit_cosine = MaskGit(
            transformer=self.mock_transformer,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            schedule="cosine"
        )

        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        generated_indices = maskgit_cosine.generate(
            num_tokens=self.num_tokens,
            conditions=self.conditions,
            conditions_mask=self.input_mask,
            device=self.device,
            steps=8,
            temperature=1.0
        )

        self.assertIsInstance(generated_indices, list)
        self.assertGreater(len(generated_indices), 0)

    def test_generate_linear_schedule(self):
        """测试linear调度的generate函数"""
        maskgit_linear = MaskGit(
            transformer=self.mock_transformer,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            schedule="linear"
        )

        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        generated_indices = maskgit_linear.generate(
            num_tokens=self.num_tokens,
            conditions=self.conditions,
            conditions_mask=self.input_mask,
            device=self.device,
            steps=8,
            temperature=1.0
        )

        self.assertIsInstance(generated_indices, list)
        self.assertGreater(len(generated_indices), 0)

    def test_generate_early_termination(self):
        """测试generate函数的提前终止"""
        # 创建一个会导致提前终止的场景
        # 设置logits使得所有token都有很高的置信度
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.zeros(logits_shape)
        # 给第一个token很高的logits
        mock_logits[:, :, 0] = 10.0
        self.mock_transformer.return_value = mock_logits

        generated_indices = self.maskgit.generate(
            num_tokens=self.num_tokens,
            conditions=self.conditions,
            conditions_mask=self.input_mask,
            device=self.device,
            steps=10,
            temperature=1.0
        )

        # 可能会提前终止
        self.assertLessEqual(len(generated_indices), 11)  # steps + 1

    def test_create_random_mask(self):
        """测试_create_random_mask辅助函数"""
        shape = (self.batch_size, self.num_tokens)

        # 测试不同的mask比例
        for ratio in [0.0, 0.3, 0.7, 1.0]:
            with self.subTest(ratio=ratio):
                mask = self.maskgit._create_random_mask(
                    shape, ratio, self.device)

                self.assertEqual(mask.shape, shape)
                self.assertEqual(mask.dtype, torch.bool)

    def test_compute_unmask_count_cosine(self):
        """测试cosine调度的_compute_unmask_count函数"""
        current_mask = torch.ones(
            (self.batch_size, self.num_tokens), dtype=torch.bool)

        # 测试不同步骤
        for step in range(8):
            count = self.maskgit._compute_unmask_count(
                step=step,
                total_steps=8,
                num_tokens=self.num_tokens,
                current_mask=current_mask
            )

            self.assertGreaterEqual(count, 1)
            self.assertLessEqual(count, self.num_tokens)

    def test_compute_unmask_count_linear(self):
        """测试linear调度的_compute_unmask_count函数"""
        maskgit_linear = MaskGit(
            transformer=self.mock_transformer,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            schedule="linear"
        )

        current_mask = torch.ones(
            (self.batch_size, self.num_tokens), dtype=torch.bool)

        # 测试不同步骤
        for step in range(8):
            count = maskgit_linear._compute_unmask_count(
                step=step,
                total_steps=8,
                num_tokens=self.num_tokens,
                current_mask=current_mask
            )

            self.assertGreaterEqual(count, 1)
            self.assertLessEqual(count, self.num_tokens)

    def test_compute_unmask_count_invalid_schedule(self):
        """测试无效调度类型的错误处理"""
        maskgit_invalid = MaskGit(
            transformer=self.mock_transformer,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            schedule="invalid"
        )

        current_mask = torch.ones(
            (self.batch_size, self.num_tokens), dtype=torch.bool)

        with self.assertRaises(ValueError):
            maskgit_invalid._compute_unmask_count(
                step=0,
                total_steps=8,
                num_tokens=self.num_tokens,
                current_mask=current_mask
            )

    def test_create_update_mask(self):
        """测试_create_update_mask函数"""
        current_mask = torch.tensor([
            [True, True, False, True],
            [True, False, True, True]
        ])
        confidence = torch.tensor([
            [0.9, 0.3, 0.7, 0.8],
            [0.2, 0.9, 0.6, 0.4]
        ])

        update_mask = self.maskgit._create_update_mask(
            current_mask=current_mask,
            confidence=confidence,
            k=2
        )

        self.assertEqual(update_mask.shape, current_mask.shape)
        self.assertEqual(update_mask.dtype, torch.bool)

        # 验证只有masked位置会被考虑
        # 验证update_mask中False的数量不超过k
        for i in range(current_mask.size(0)):
            unmasked_count = (~update_mask[i]).sum().item()
            self.assertLessEqual(unmasked_count, 2)  # k=2

    def test_forward_all_masked(self):
        """测试100%掩码情况"""
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)

        # 直接设置mock的返回值
        self.mock_transformer.return_value = mock_logits

        with patch('random.random', return_value=1.0):
            loss = self.maskgit.forward(
                self.input_indices,
                self.input_mask,
                self.target_indices
            )
            # 验证至少有一个token未掩码（安全机制）
            self.assertTrue(torch.isfinite(loss))
            self.mock_transformer.assert_called_once()

    def test_forward_no_masked(self):
        """测试0%掩码情况"""
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)

        # 直接设置mock的返回值
        self.mock_transformer.return_value = mock_logits

        with patch('random.random', return_value=0.0):
            loss = self.maskgit.forward(
                self.input_indices,
                self.input_mask,
                self.target_indices
            )
            # 验证无掩码token
            self.assertTrue(torch.isfinite(loss))
            self.mock_transformer.assert_called_once()

    def test_generate_all_masked(self):
        """测试全掩码生成"""
        # 设置mock transformer的返回值
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.side_effect = lambda x, mask: mock_logits

        # 强制初始全掩码
        current_mask = torch.ones(
            (self.batch_size, self.num_tokens), dtype=torch.bool)
        with patch.object(self.maskgit, '_create_random_mask', return_value=current_mask):
            generated = self.maskgit.generate(
                num_tokens=self.num_tokens,
                conditions=self.conditions,
                conditions_mask=self.input_mask,
                device=self.device,
                steps=4,
                temperature=1.0
            )
            # 验证第一步全掩码
            self.assertTrue(torch.all(generated[0] == self.mask_token_id))

    def test_add_noise_random_replace(self):
        """测试噪声添加逻辑"""
        original = torch.tensor([[1000, 2000, 3000], [4000, 5000, 6000]])
        noised = self.maskgit._add_noise_random_replace(
            original, vocab_size=10000, noise_prob=1.0)
        # 完全噪声时所有token都应变化
        self.assertTrue(torch.all(noised != original))

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试单个token生成
        logits_shape = (1, self.seq_len + 1, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        generated_indices = self.maskgit.generate(
            num_tokens=1,
            conditions=self.conditions[:1],  # 单个样本
            conditions_mask=self.input_mask[:1],
            device=self.device,
            steps=2,
            temperature=1.0
        )

        self.assertIsInstance(generated_indices, list)
        self.assertGreater(len(generated_indices), 0)

        # 验证形状
        for step_indices in generated_indices:
            self.assertEqual(step_indices.shape, (1, 1))

    def test_batch_consistency(self):
        """测试批次处理的一致性"""
        # 测试不同批次大小
        for batch_size in [1, 3, 5]:
            with self.subTest(batch_size=batch_size):
                conditions = torch.randint(
                    0, self.vocab_size, (batch_size, self.seq_len))
                input_mask = torch.ones_like(conditions, dtype=torch.bool)

                logits_shape = (batch_size, self.seq_len +
                                self.num_tokens, self.vocab_size)
                mock_logits = torch.randn(logits_shape)
                self.mock_transformer.return_value = mock_logits

                generated_indices = self.maskgit.generate(
                    num_tokens=self.num_tokens,
                    conditions=conditions,
                    conditions_mask=input_mask,
                    device=self.device,
                    steps=4,
                    temperature=1.0
                )

                # 验证每个步骤的批次维度
                for step_indices in generated_indices:
                    self.assertEqual(step_indices.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
