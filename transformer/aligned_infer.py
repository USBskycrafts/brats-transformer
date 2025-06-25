import math
import random
from logging import warning
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from transformers import Dinov2Model

from autoencoder.vqvae import VQVAE
from transformer.transformer import TransformerEncoderModel


class MultiContrastGenerationInferer(nn.Module):
    aligned_weight: int

    def __init__(
        self,
        spatial_dims: int = 2,
        latent_dims: int = 5,  # latent dim from vae
        latent_size: int | Sequence[int] = 12,
        hidden_dim: int = 512,
        num_contrasts: int = 4,
        max_token_len: int = 1024,
        pretrained_dir: str = 'facebook/dinov2-base',
        aligned_weight: int = 0.5
    ):
        super().__init__()
        self.aligned_weight = aligned_weight

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

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_token_len, hidden_dim)
        )

        # 可学习的mask token embedding
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )

        self.dinov2 = Dinov2Model.from_pretrained(pretrained_dir)
        for p in self.dinov2.parameters():
            p.requires_grad = False

    def _align_hidden_state(self, p, q, indices):
        q = rearrange(q, 'b (h w) c -> b c h w', h=int(math.sqrt(q.shape[1])),
                      w=int(math.sqrt(q.shape[1])))
        p = rearrange(p, 'b (h w) c -> b c h w', h=int(math.sqrt(p.shape[1])),
                      w=int(math.sqrt(p.shape[1])))
        q = F.interpolate(
            q,
            p.shape[2:],
            mode='bilinear'
        )
        p = rearrange(p, 'b c h w -> b (h w) c')
        q = rearrange(q, 'b c h w -> b (h w) c')
        p = p.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, indices.size(-1))
        )
        q = q.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, indices.size(-1))
        )
        return F.mse_loss(p, q)

    def forward(self,
                imgs,
                input_contrasts,
                target,
                target_contrasts,
                vqvae: VQVAE,
                transformer: TransformerEncoderModel
                ):
        features = []
        for img, input_contrast in zip(imgs, input_contrasts.split(1, dim=1)):
            features.append(self.contrast_embedding[input_contrast])
            with torch.no_grad():
                feature = vqvae.encode_stage_2_inputs(img)

            feature = self.patch_embed(feature)
            features.append(feature)

        # 添加维度使contrast embedding可以连接
        # (B, 1, hidden_dim)
        contrast_emb = self.contrast_embedding[target_contrasts]
        features.append(contrast_emb)

        with torch.no_grad():
            feature = vqvae.encode_stage_2_inputs(target)
        feature = self.patch_embed(feature)
        features.append(feature)

        # generate the masked token for target inputs
        with torch.no_grad():
            indices = vqvae.index_quantize(target)
            indices = indices.flatten(1)
            mask_ratio = math.cos(
                random.random() * math.pi / 2
            )
            num_to_mask = max(1, int(
                mask_ratio * indices.shape[1]
            ))
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
        features = features + self.pos_embedding[:, :features.shape[1], :]

        spatial_dims = len(target.shape[2:])
        if spatial_dims == 2:
            with torch.no_grad():
                target_bar = target.repeat(
                    (1, 3, 1, 1) if spatial_dims == 2 else (1, 3, 1, 1, 1))
                target_bar = F.interpolate(
                    target_bar,
                    (224, 224),
                    mode='bilinear',
                )
                target_feature = self.dinov2(target_bar).last_hidden_state
                target_feature = target_feature[:, :-1, :]
            last_hidden_state, logits = transformer(features)
            last_hidden_state = last_hidden_state[:, -indices.shape[1]:]
            dloss = self._align_hidden_state(
                last_hidden_state,
                target_feature,
                mask_indices
            )
        else:
            warning('dinov2 does not support 3D images yet')
            dloss = 0.0
            _, logits = transformer(features)

        # select the logits and labels for masked tokens
        logits_masked = logits[:, -indices.shape[1]:].gather(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        )
        labels_masked = indices.gather(
            1,
            mask_indices
        )
        # caculate loss
        closs = self.loss(logits_masked.transpose(1, -1),
                          labels_masked)
        return closs, dloss, logits_masked, labels_masked

    @torch.no_grad()
    def sample(self,
               imgs,
               input_contrasts,
               target_contrasts,
               vqvae: VQVAE,
               transformer: TransformerEncoderModel,
               iterations: int = 6
               ):
        device = imgs[0].device

        # 提取输入图像特征
        features = []
        spatial_sizes = []
        for img, input_contrast in zip(imgs, input_contrasts.split(1, dim=1)):
            features.append(self.contrast_embedding[input_contrast])
            feature = vqvae.encode_stage_2_inputs(img)
            spatial_sizes.append(feature.shape[2:])
            feature = self.patch_embed(feature)
            features.append(feature)
        assert all(s == spatial_sizes[0] for s in spatial_sizes), \
            "All input images must have the same spatial size."
        # 添加维度使contrast embedding可以连接
        # (B, 1, hidden_dim)
        contrast_emb = self.contrast_embedding[target_contrasts]
        features.append(contrast_emb)
        features = torch.cat(features, dim=1)

        # number of tokens to predict
        num_predict_tokens = math.prod(spatial_sizes[0])
        predicted_indices = torch.full(
            (features.size(0), num_predict_tokens),
            fill_value=-1,
            device=device
        )
        masked_tokens = self.mask_token.expand(
            features.size(0), num_predict_tokens, features.size(-1)
        )
        mask = (predicted_indices == -1)
        torch.set_printoptions(profile="full")
        for i in range(iterations):
            input_tokens = torch.cat(
                [features, masked_tokens], dim=1
            )
            input_tokens = input_tokens + \
                self.pos_embedding[:, :input_tokens.shape[1], :]
            _, logits = transformer(input_tokens)
            logits = logits[:, -num_predict_tokens:]

            prob = F.softmax(logits, dim=-1)
            indices = torch.distributions.Categorical(probs=prob).sample()
            conf = torch.gather(
                prob, dim=-1, index=indices.unsqueeze(-1)
            ).squeeze(-1)

            conf = torch.randn_like(conf) if i < 2 else conf
            conf[~mask] = math.inf

            t = 1 - math.cos(
                (i + 1) / iterations * math.pi / 2
            )
            tresh_conf, index_mask = conf.topk(
                k=round((t * num_predict_tokens)),
                dim=-1
            )
            tresh_conf = tresh_conf[:, [-1]]
            f_mask = (conf >= tresh_conf)
            predicted_indices[f_mask] = indices[f_mask]

            mask = (predicted_indices == -1)
            # print(f"Iteration {i + 1}/{iterations}, "
            #       f"Masked tokens: {mask.sum().item()}/{num_predict_tokens}")
            if mask.sum() == 0:
                break

            spatial_dims = spatial_sizes[0]
            src_indices = indices.reshape(
                indices.size(0), *spatial_dims
            )
            src_features = vqvae.quantizer.embed(src_indices)
            src_features = self.patch_embed(src_features)

            masked_tokens = masked_tokens.clone()
            batch_indices = torch.arange(
                masked_tokens.shape[0]).unsqueeze(1).expand_as(index_mask)
            masked_tokens[batch_indices, index_mask
                          ] = src_features[batch_indices, index_mask]

        spatial_dims = spatial_sizes[0]
        predicted_indices = predicted_indices.reshape(
            predicted_indices.size(0), *spatial_dims
        )

        predict = vqvae.decode_samples(predicted_indices)
        return predict.clamp(-1, 1)
