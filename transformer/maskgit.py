# File: transformer/maskgit.py
# -*- coding: utf-8 -*-
# @Author: Junran Yang, DeepSeek
# @Date: 2025-06-27
# @Last Modified by: Junran Yang, DeepSeek
# @Last Modified time: 2025-06-27
# @Description: Masked Generative Image Transformer (MaskGit) model implementation.

import random
from sympy import false
import torch
import torch.nn as nn
import math
from typing import Tuple, Union, Optional

from autoencoder.modules import VQVAE


class MaskGit(nn.Module):
    """    Masked Generative Image Transformer (MaskGit) model.
    Args:
        transformer (nn.Module): The transformer model.
        vqvae (nn.Module): The VQ-VAE model.
        mask_token_id (int): The token ID used for masking.
        scheduler (str, optional): The learning rate scheduler type. Defaults to "cosine".
    """
    transformer: nn.Module
    mask_token_id: int
    vocab_size: int
    schedule: str

    def __init__(
        self,
        transformer: nn.Module,
        mask_token_id: int,
        vocab_size: int,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.transformer = transformer
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.schedule = schedule

    def forward(
        self,
        input_indices: torch.Tensor,
        target_indices: torch.Tensor
    ):
        """
        Args:
            indices: (B, L) tensor of indices to be transformed.
            mask: (B, L) tensor of boolean values indicating masked positions.
        Returns:
            logits: (B, L, vocab_size) tensor of logits for each index.
            input_mask: (B, L) tensor of boolean values indicating masked positions.(True means masked)
        """
        mask_ratio = random.random()
        target_mask = self._create_random_mask(
            target_indices.shape,
            mask_ratio,
            device=target_indices.device
        )

        # generate the masked input indices
        masked_indices = target_indices.where(
            target_mask,
            self.mask_token_id
        )
        # calculate the logits from the transformer and return
        logits = self.transformer(
            torch.cat(
                [input_indices, masked_indices],
                dim=1
            )
        )

        target_logits = logits[:, -target_indices.size(1):]
        target_logits = target_logits[~target_mask, :]
        target_indices = target_indices[~target_mask]
        return torch.nn.functional.cross_entropy(
            target_logits,
            target_indices
        )

    @torch.no_grad()
    def generate(
        self,
        num_tokens: int,
        conditions: torch.Tensor,
        device: torch.device,
        steps: int = 8,
        temperature: float = 1.0,
    ):
        """
        Generate new tokens based on the conditions.
        Args:
            num_tokens: Number of tokens to generate.
            conditions: (B, L) tensor of condition indices.
            device: The device to run the model on.
            steps: Number of generation steps.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            threshold: Threshold for filtering logits.
        Returns:
            generated_indices: list of (B, num_tokens) tensor of generated indices.
        """
        batch_size, seq_len = conditions.shape

        # Initialize the current tokens with mask token ID and mask
        current_indices = torch.full(
            (batch_size, num_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        current_mask = torch.ones_like(
            current_indices,
            dtype=torch.bool,
            device=device
        )

        # save the intermediate generated indices
        generated_indices = [current_indices]
        for step in range(1, steps + 1):
            if current_mask.sum() == 0:
                # If no tokens are masked, break the loop
                break

            input_indices = torch.cat(
                (conditions, current_indices),
                dim=1
            )
            logits = self.transformer(input_indices)[:, -num_tokens:]
            probs = torch.softmax(logits / temperature, dim=-1)
            max_probs, max_indices = probs.max(dim=-1)

            # Combine conditions and current tokens
            k = self._compute_unmask_count(
                step,
                steps,
                num_tokens,
                current_mask
            )
            update_mask = self._create_update_mask(
                current_mask,
                max_probs,
                k
            )
            current_mask = current_mask & update_mask
            current_indices = torch.where(
                update_mask,
                current_indices,
                max_indices
            )
            generated_indices.append(current_indices)

        return generated_indices

    def _create_random_mask(
        self,
        shape: Tuple[int, ...],
        mask_ratio: float,
        device: torch.device
    ):
        mask = torch.rand(shape, device=device) < mask_ratio
        return mask

    def _compute_unmask_count(
        self,
        step: int,
        total_steps: int,
        num_tokens: int,
        current_mask: torch.Tensor
    ) -> int:
        """
        Compute the number of unmasked tokens based on the current step.
        Args:
            step: Current generation step.
            total_steps: Total number of steps.
            num_tokens: Total number of tokens to generate.
            current_mask: Current mask tensor.
        Returns:
            unmask_count: Number of tokens to unmask in the current step. 
        """
        if self.schedule == 'cosine':
            ratio = math.cos((step / total_steps) * math.pi / 2)
            unmask_ratio = 1.0 - ratio
            k = max(1, int(num_tokens * unmask_ratio))
        elif self.schedule == 'linear':
            ratio = step / total_steps
            unmask_ratio = 1.0 - ratio
            k = max(1, int(num_tokens * unmask_ratio))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

        # Ensure we do not unmask more tokens than currently masked
        max_masked_count = int(current_mask.sum(dim=1).max().item())
        return min(k, max_masked_count)

    def _create_update_mask(
        self,
        current_mask: torch.Tensor,
        confidence: torch.Tensor,
        k: int,
    ):
        """
        Create a mask for updating tokens based on confidence scores.
        Args:
            current_mask: (B, L) Current mask tensor.
            confidence: (B, L) Confidence scores for each token.
            k: Number of tokens to update per sample.
        Returns:
            update_mask: (B, L) Mask indicating which tokens to update.
        """
        # Create mask where only masked positions have confidence scores
        valid_confidence = torch.where(
            current_mask,
            confidence,
            torch.full_like(confidence, float('-inf'))
        )

        # Get top k confident tokens per sample
        _, top_indices = torch.topk(
            valid_confidence,
            k=k,
            dim=1  # Get top k along sequence dimension
        )

        # Create update mask (False means unmask)
        update_mask = torch.ones_like(current_mask, dtype=torch.bool)

        # Unmask the top k tokens for each sample in batch
        batch_indices = torch.arange(current_mask.size(
            0), device=current_mask.device)[:, None]
        update_mask[batch_indices, top_indices] = False

        return update_mask
