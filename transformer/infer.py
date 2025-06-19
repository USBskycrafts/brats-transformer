import math
import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from numpy import mask_indices

from autoencoder.vqvae import VQVAE
from transformer.transformer import ContrastGenerationTransformer


class MultiContrastGenerationInferer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        latent_dims: int = 5,  # latent dim from vae
        latent_size: int | Sequence[int] = 12,
        hidden_dim: int = 512,
        num_contrasts: int = 4
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.patch_embed = PatchEmbeddingBlock(
            spatial_dims=spatial_dims,
            in_channels=latent_dims,
            img_size=latent_size,
            patch_size=1,
            num_heads=8,
            hidden_size=hidden_dim,
            proj_type="conv",
            pos_embed_type="none"
        )

        self.contrast_embedding = nn.Parameter(
            torch.randn(num_contrasts, hidden_dim)
        )

        # 可学习的mask token embedding
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )

        if isinstance(latent_size, int):
            num_predict_tokens = latent_size ** spatial_dims
        else:
            num_predict_tokens = math.prod(latent_size)
        self.num_predict_tokens = num_predict_tokens

    def forward(self,
                imgs,
                contrasts,
                target,
                vqvae: VQVAE,
                transformer: ContrastGenerationTransformer
                ):
        features = []
        for img in imgs:
            # with torch.no_grad():
            feature = vqvae.encode_stage_2_inputs(img)

            feature = self.patch_embed(feature)
            features.append(feature)

        # 添加维度使contrast embedding可以连接
        contrast_emb = self.contrast_embedding[contrasts]  # (B, 1, hidden_dim)
        features.append(contrast_emb)

        # with torch.no_grad():
        feature = vqvae.encode_stage_2_inputs(target)

        feature = self.patch_embed(feature)
        features.append(feature)

        # with torch.no_grad():
        indices = vqvae.index_quantize(target)
        indices = indices.flatten(1)
        mask_ratio = math.cos(
            random.random() * math.pi / 2
        )
        num_to_mask = int(
            mask_ratio * indices.shape[1]
        )
        rand_indices = torch.rand(
            indices.shape, device=indices.device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_to_mask]
        batch_size, seq_len, feature_dim = features[-1].shape
        _, num_to_mask = mask_indices.shape
        index_to_scatter = mask_indices.unsqueeze(
            -1).expand(-1, -1, feature_dim)
        src = self.mask_token.expand(batch_size, num_to_mask, feature_dim)
        masked_input = features[-1].clone()
        masked_input.scatter_(dim=1, index=index_to_scatter, src=src)
        features[-1] = masked_input

        features = torch.cat(features, dim=1)

        # unmask the tokens
        logits = transformer(features)

        logits_masked = logits[:, -indices.shape[1]:].gather(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        )
        labels_masked = indices.gather(
            1,
            mask_indices
        )
        # caculate loss
        loss = self.loss(logits_masked.transpose(1, -1),
                         labels_masked)
        return loss, logits_masked, labels_masked

    @torch.no_grad()
    def sample(self,
               imgs,
               contrasts,
               vqvae: VQVAE,
               transformer: ContrastGenerationTransformer,
               iterations: int = 8
               ):
        device = next(transformer.parameters()).device

        # 提取输入图像特征
        features = []
        for img in imgs:
            feature = vqvae.encode_stage_2_inputs(img)
            feature = self.patch_embed(feature)
            features.append(feature)
        # 添加维度使contrast embedding可以连接
        contrast_emb = self.contrast_embedding[contrasts]  # (B, 1, hidden_dim)
        features.append(contrast_emb)
        features = torch.cat(features, dim=1)

        # 初始化候选张量 - 使用可学习的mask token
        num_predict_tokens = self.num_predict_tokens
        candidate = self.mask_token.expand(
            features.size(0), num_predict_tokens, -1
        ).clone()

        # 跟踪哪些tokens还是masked状态
        mask_schedule = torch.ones(
            (features.size(0), num_predict_tokens),
            dtype=torch.bool,
            device=device
        )

        for i in range(iterations):
            # 通过transformer获取logits
            logits = transformer(
                torch.cat([features, candidate], dim=1)
            )

            # 提取预测tokens的logits
            predict_logits = logits[:, -num_predict_tokens:]

            # 计算概率分布和置信度
            probs = F.softmax(predict_logits, dim=-1)
            confidence = probs.max(dim=-1)[0]  # 最大概率作为置信度

            # 获取预测的token indices
            predicted_tokens = probs.argmax(dim=-1)

            # 计算这次迭代要unmask的token数量
            # 使用cosine schedule: 开始unmask少量，逐渐增加
            ratio = (i + 1) / iterations
            num_to_unmask = int(ratio * num_predict_tokens)

            if i == iterations - 1:
                # 最后一次迭代，unmask所有剩余的tokens
                tokens_to_unmask = mask_schedule
            else:
                # 选择置信度最高的masked tokens进行unmask
                masked_confidence = confidence.clone()
                masked_confidence[~mask_schedule] = -1  # 已unmask的位置设为最低置信度

                # 获取top-k最高置信度的token位置
                _, top_indices = masked_confidence.topk(
                    min(num_to_unmask, mask_schedule.sum(dim=1).max().item()),
                    dim=1
                )

                # 创建unmask掩码
                seq_len = mask_schedule.size(1)
                tokens_to_unmask = torch.zeros_like(
                    mask_schedule, dtype=torch.bool)
                valid_indices_mask = top_indices < seq_len
                safe_indices = torch.where(valid_indices_mask, top_indices, 0)
                src_values = valid_indices_mask
                tokens_to_unmask.scatter_(
                    dim=1, index=safe_indices, src=src_values)

                # 只unmask当前还是masked状态的tokens
                tokens_to_unmask = tokens_to_unmask & mask_schedule

            # 更新候选张量
            if tokens_to_unmask.any():
                # 从输入图像获取实际的latent空间形状
                with torch.no_grad():
                    sample_feature = vqvae.encode_stage_2_inputs(imgs[0])
                spatial_shape = sample_feature.shape[2:]  # 获取空间维度

                predicted_tokens_reshaped = predicted_tokens.view(
                    predicted_tokens.size(0), *spatial_shape
                )

                # 将预测的token indices转换为embedding
                predicted_embeddings = vqvae.quantizer.embed(
                    predicted_tokens_reshaped)
                predicted_embeddings = self.patch_embed(predicted_embeddings)

                # 更新candidate tensor中被选中的位置
                for batch_idx in range(candidate.size(0)):
                    unmask_positions = tokens_to_unmask[batch_idx]
                    if unmask_positions.any():
                        candidate[batch_idx,
                                  unmask_positions] = predicted_embeddings[batch_idx, unmask_positions]

                # 更新mask schedule
                mask_schedule = mask_schedule & (~tokens_to_unmask)

        # 最终解码：从最后的logits获取token indices
        final_logits = transformer(torch.cat([features, candidate], dim=1))
        final_predict_logits = final_logits[:, -num_predict_tokens:]
        final_tokens = final_predict_logits.argmax(dim=-1)

        # 从输入图像获取实际的latent空间形状进行重塑
        with torch.no_grad():
            sample_feature = vqvae.encode_stage_2_inputs(imgs[0])
        spatial_shape = sample_feature.shape[2:]  # 获取空间维度

        final_tokens_reshaped = final_tokens.view(
            final_tokens.size(0), *spatial_shape
        )

        # 通过VQ-VAE解码生成最终图像
        decoded_latents = vqvae.quantizer.embed(final_tokens_reshaped)
        generated_images = vqvae.decode_stage_2_outputs(decoded_latents)

        return generated_images
