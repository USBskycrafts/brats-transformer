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
                 num_embeddings: int = 1024,
                 embedding_dim: int = 3,
                 act: tuple | str | None = "SWISH",
                 num_contrast: int = 4,
                 patch_size: int = 2,
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 lr=1.0e-4,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.infer = MultiContrastGenerationInferer()
        self.vqvae = VQVAE(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
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

        self.transformer = ContrastGenerationTransformer(
            in_channels=embedding_dim,
            img_size=img_size,
            spatial_dims=spatial_dims,
            num_contrast=num_contrast,
            hidden_size=hidden_size,
            patch_size=patch_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_embeddings=num_embeddings,
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

        psnr = PSNR(max_val=1.0)(generated, target)
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 2),
                    data_range=1.0)(generated, target)
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
        psnr = PSNR(max_val=1.0)(generated, target)
        # 计算SSIM
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 2),
                    data_range=1.0)(generated, target)
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
        optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.hparams.get('lr', 1.0e-4),
            betas=(0.9, 0.95)
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
