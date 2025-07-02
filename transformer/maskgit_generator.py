import math
from typing import Sequence, Union, Dict

from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from monai.losses.perceptual import PerceptualLoss as LPIPS
from monai.metrics.regression import PSNRMetric as PSNR
from monai.metrics.regression import SSIMMetric as SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR

from autoencoder.modules import VQVAE
from transformer.maskgit import MaskGit
from transformer.transformer import TransformerEncoderModel


class ContrastMaskGiT(pl.LightningModule):
    def __init__(self,
                 spatial_dims: int,
                 img_size: Sequence[int] | int,
                 stage_one_ckpt: str,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 channels: Sequence[int] = (32, 32, 64, 128, 256),
                 num_res_layers: int = 2,
                 num_res_channels: Sequence[int] | int = (
                     32, 32, 64, 128, 256),
                 downsample_parameters: Sequence[tuple] = (
                     (0, 7, 1, 3),
                     (1, 4, 1, 1),
                     (1, 4, 1, 1),
                     (1, 4, 1, 1),
                     (1, 4, 1, 1),
                 ),
                 upsample_parameters: Sequence[tuple] = (
                     (1, 4, 1, 1, 0),
                     (1, 4, 1, 1, 0),
                     (1, 4, 1, 1, 0),
                     (1, 4, 1, 1, 0),
                     (0, 7, 1, 3, 0),
                 ),
                 embedding_dim: int = 5,
                 levels: Sequence[int] = (8, 8, 8, 6, 5),
                 act: tuple | str | None = "SWISH",
                 num_contrast: int = 4,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 lr=1.0e-4,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # 首先创建 VQ-VAE 来确定 latent 尺寸
        self.vqvae = VQVAE(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            embedding_dim=embedding_dim,
            levels=levels,
            act=act,
        )
        ckpt = torch.load(stage_one_ckpt, map_location='cpu')['state_dict']
        ckpt = {
            k.replace('vqvae.', ''): v for k, v in ckpt.items()
            if k.startswith('vqvae.')
        }
        self.vqvae.load_state_dict(
            ckpt, strict=True
        )
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

        vocab_size = math.prod(levels) + 1 + num_contrast
        self.transformer = TransformerEncoderModel(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_embeddings=vocab_size,
        )

        self.maskgit = MaskGit(
            transformer=self.transformer,
            # +1 for mask token, +num_contrast for contrast embeddings
            vocab_size=vocab_size,
            mask_token_id=math.prod(levels),
        )

        self.visual_proj = nn.Linear(
            hidden_size,
            embedding_dim
        )

        self.contrast_offset = math.prod(levels) + 1
        self.img_shape = [img_size] * \
            spatial_dims if isinstance(img_size, int) else img_size

    def forward(self, imgs, input_contrasts, target, target_contrasts):
        input_indices = []
        input_mask = []

        # generate the target contrast prompt
        modality = target_contrasts + self.contrast_offset
        input_indices.append(modality - 1)

        modality_mask = torch.ones_like(modality)
        input_mask.append(modality_mask)

        # generate the input indices
        for img, modality in zip(imgs, input_contrasts.split(1, dim=1)):
            indices = self.vqvae.index_quantize(img)
            input_indices.append(indices.flatten(1))
            modality = modality + self.contrast_offset
            input_indices.append(modality - 1)

            # generate padding mask
            modality_mask = (modality != 0)
            batch_size, seq_len = indices.flatten(1).shape
            modality_mask = modality_mask.expand(
                -1,
                seq_len + 1
            )
            input_mask.append(modality_mask)

        # now generate the sequence
        input_indices = torch.cat(input_indices, dim=1)
        input_mask = torch.cat(input_mask, dim=1).type(torch.bool)
        assert input_indices.shape == input_mask.shape, f"{input_indices.shape} and {input_mask.shape} should have the same shape"
        if target is not None:
            target_indices = self.vqvae.index_quantize(target)
            target_indices = target_indices.flatten(1)
            target_features = self.vqvae.encode(target)
            target_features = rearrange(
                target_features,
                'b c h w -> b (h w) c'
            )
            return input_indices, input_mask, target_indices, target_features
        else:
            return input_indices, input_mask

    def training_step(self, batch, batch_idx):
        imgs, input_contrasts, target, target_contrasts = self._prepare_input(
            batch)
        with torch.no_grad():
            input_indices, input_mask, target_indices, target_features = self(
                imgs, input_contrasts, target, target_contrasts)

        embeds = self.transformer.embedding(target_indices)
        mse_loss = F.mse_loss(
            self.visual_proj(embeds),
            target_features
        )

        # caculate the loss
        ce_loss = self.maskgit(
            input_indices, input_mask, target_indices
        )

        self.log('train/ce_loss', ce_loss, on_step=True, sync_dist=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mse_loss', mse_loss, on_step=True, sync_dist=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return ce_loss + 0.5 * mse_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, input_contrasts, target, target_contrasts = self._prepare_input(
            batch)
        input_indices, input_mask, target_indices, *_ = self(
            imgs, input_contrasts, target, target_contrasts)
        generated_indices = self.maskgit.generate(
            num_tokens=target_indices.size(1),
            conditions=input_indices,
            conditions_mask=input_mask,
            device=input_indices.device
        )
        last_indices = generated_indices[-1]
        latent_indices = last_indices.view(-1, *self.img_shape)
        generated = self.vqvae.decode_samples(latent_indices).clamp(-1, 1)

        psnr = PSNR(max_val=2.0)(generated, target)
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 2),
                    data_range=2.0)(generated, target)
        lpips = LPIPS(
            spatial_dims=self.hparams.get('spatial_dims', 2)
        ).to(self.device)(generated, target)

        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)
        self.log('val/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('val/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('val/ploss', lpips, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {
            'input': torch.cat(imgs, dim=0),
            'generated': generated,
            'target': target,
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, input_contrasts, target, target_contrasts = self._prepare_input(
            batch)
        input_indices, input_mask, target_indices, *_ = self(
            imgs, input_contrasts, target, target_contrasts)
        generated_indices = self.maskgit.generate(
            num_tokens=target_indices.size(1),
            conditions=input_indices,
            conditions_mask=input_mask,
            device=input_indices.device
        )
        last_indices = generated_indices[-1]
        latent_indices = last_indices.view(-1, *self.img_shape)
        generated = self.vqvae.decode_samples(latent_indices)

        # 计算PSNR
        psnr = PSNR(max_val=2.0)(generated, target)
        # 计算SSIM
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 2),
                    data_range=2.0)(generated, target)
        lpips = LPIPS(
            spatial_dims=self.hparams.get('spatial_dims', 2)
        ).to(self.device)(generated, target)
        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)

        self.log('test/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test/ploss', lpips, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return {
            'input': torch.cat(imgs, dim=0),
            'generated': generated,
            'target': target,
        }

    def configure_optimizers(self):  # type: ignore
        # 收集所有需要优化的参数
        params_to_optimize = []

        # 添加 transformer 参数
        params_to_optimize.extend(self.transformer.parameters())

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.hparams.get('lr', 1.0e-4),
            betas=(0.9, 0.95),
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(step / 2000, 1.0)
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': warmup_scheduler,
                'interval': 'step'
            }
        }
    # -------------------------------------------------------------------
    # private methods

    @torch.no_grad()
    def _prepare_input(self, batch):
        # support 2d and 3d
        imgs, input_contrasts, target, target_contrasts = batch
        if not isinstance(imgs, list):
            imgs = torch.chunk(imgs,
                               chunks=self.hparams.get('num_contrast', 0) - 1,
                               dim=1)
        imgs = [img.to(self.device) for img in imgs]
        input_contrasts = input_contrasts.to(self.device)
        target_contrasts = target_contrasts.to(self.device)
        target = target.to(self.device)
        return imgs, input_contrasts, target, target_contrasts
