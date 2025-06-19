import math
from typing import Sequence

import pytorch_lightning as pl
import torch
from monai.losses.perceptual import PerceptualLoss as LPIPS
from monai.metrics.regression import PSNRMetric as PSNR
from monai.metrics.regression import SSIMMetric as SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR

from autoencoder.modules import VQVAE
from transformer.infer import MultiContrastGenerationInferer
from transformer.transformer import ContrastGenerationTransformer
import pytorch_optimizer


class ContrastGeneration(pl.LightningModule):
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

        # 创建 MultiContrastGenerationInferer 并传递正确的参数
        self.infer = MultiContrastGenerationInferer(
            spatial_dims=spatial_dims,
            latent_dims=embedding_dim,
            latent_size=img_size,
            hidden_dim=hidden_size,
            num_contrasts=num_contrast
        )

        self.transformer = ContrastGenerationTransformer(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_embeddings=math.prod(levels),
        )

    def forward(self, imgs, contrasts, target):
        return self.infer(imgs,
                          contrasts,
                          target,
                          self.vqvae,
                          self.transformer
                          )

    def training_step(self, batch, batch_idx):
        imgs, contrasts, target = self._prepare_input(batch)
        loss, *_ = self.infer(
            imgs,
            contrasts,
            target,
            self.vqvae,
            self.transformer
        )
        self.log('train/loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, contrasts, target = self._prepare_input(batch)
        generated = self.infer.sample(
            imgs,
            contrasts,
            self.vqvae,
            self.transformer
        )

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
        # 准备输入数据
        imgs, contrasts, target = self._prepare_input(batch)
        # 生成图像
        generated = self.infer.sample(
            imgs,
            contrasts,
            self.vqvae,
            self.transformer
        )

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

    def configure_optimizers(self):
        # 收集所有需要优化的参数
        params_to_optimize = []

        # 添加 transformer 参数
        params_to_optimize.extend(self.transformer.parameters())

        # 添加 inferer 的可学习参数 (mask_token, contrast_embedding 等)
        params_to_optimize.extend(self.infer.parameters())

        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.hparams.get('lr', 1.0e-4),
        )
        return [optimizer], []
    # -------------------------------------------------------------------
    # private methods

    @torch.no_grad()
    def _prepare_input(self, batch):
        # support 2d and 3d
        imgs, contrasts, target = batch
        if not isinstance(imgs, list):
            imgs = torch.chunk(imgs,
                               chunks=self.hparams.get('num_contrast', 0) - 1,
                               dim=1)
        imgs = [img.to(self.device) for img in imgs]
        contrasts = contrasts.to(self.device)
        target = target.to(self.device)
        return imgs, contrasts, target
