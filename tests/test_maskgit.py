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

    def test_forward_basic(self):
        """测试forward函数的基本功能"""
        # 设置mock transformer的返回值
        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        # 调用forward函数
        loss = self.maskgit.forward(self.input_indices, self.target_indices)

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
        self.mock_transformer.return_value = mock_logits

        # 测试不同的mask比例
        test_ratios = [0.1, 0.5, 0.9]

        for ratio in test_ratios:
            with self.subTest(mask_ratio=ratio):
                mock_random.return_value = ratio

                loss = self.maskgit.forward(
                    self.input_indices, self.target_indices)

                # 验证loss是有限的
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(loss.item(), 0)  # 交叉熵loss应该为正

    def test_forward_mask_token_replacement(self):
        """测试forward函数中mask token的替换"""
        # 创建特殊的target_indices来验证mask
        special_target = torch.full((self.batch_size, self.num_tokens), 100)

        logits_shape = (self.batch_size, self.seq_len +
                        self.num_tokens, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        with patch('random.random', return_value=0.5):  # 50% mask ratio
            loss = self.maskgit.forward(self.input_indices, special_target)

            # 验证transformer被调用
            self.mock_transformer.assert_called()

            # 验证输入包含mask token
            call_args = self.mock_transformer.call_args[0][0]
            target_part = call_args[:, -self.num_tokens:]

            # 应该有一些位置被替换为mask_token_id
            self.assertTrue(torch.any(target_part == self.mask_token_id))

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

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试单个token生成
        logits_shape = (1, self.seq_len + 1, self.vocab_size)
        mock_logits = torch.randn(logits_shape)
        self.mock_transformer.return_value = mock_logits

        generated_indices = self.maskgit.generate(
            num_tokens=1,
            conditions=self.conditions[:1],  # 单个样本
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

                logits_shape = (batch_size, self.seq_len +
                                self.num_tokens, self.vocab_size)
                mock_logits = torch.randn(logits_shape)
                self.mock_transformer.return_value = mock_logits

                generated_indices = self.maskgit.generate(
                    num_tokens=self.num_tokens,
                    conditions=conditions,
                    device=self.device,
                    steps=4,
                    temperature=1.0
                )

                # 验证每个步骤的批次维度
                for step_indices in generated_indices:
                    self.assertEqual(step_indices.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
