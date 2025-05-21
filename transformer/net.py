from typing import Sequence

import pytorch_lightning as pl
import torch
from monai.metrics.regression import MultiScaleSSIMMetric as mSSIM
from monai.metrics.regression import PSNRMetric as PSNR
from monai.metrics.regression import SSIMMetric as SSIM
from monai.networks.nets.vqvae import VQVAE

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
                 hidden_size: int = 512,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 num_heads: int = 8,
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
        self.vqvae.load_state_dict(
            torch.load(stage_one_ckpt, map_location='cpu'),
            strict=True
        )
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

        self.transformer = ContrastGenerationTransformer(
            in_channels=embedding_dim,
            img_size=img_size,
            num_contrast=num_contrast,
            hidden_size=hidden_size,
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

    def validation_step(self, batch, batch_idx):
        imgs, contrasts, target = self._prepare_input(batch)
        loss, y, indices = self.infer(
            imgs,
            contrasts,
            target,
            self.vqvae,
            self.transformer
        )

        psnr = PSNR(max_val=1.0)(y, target)
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 1),
                    data_range=1.0)(y, target)
        m_ssim = mSSIM(spatial_dims=self.hparams.get('spatial_dims', 1),
                       data_range=1.0)(y, target)
        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)
        assert isinstance(m_ssim, torch.Tensor)

        self.log('val/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('val/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val/m_ssim', m_ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {
            'input': imgs,
            'generated': y,
            'target': target,
        }

    def test_step(self, batch, batch_idx):
        imgs, contrasts, target = self._prepare_input(batch)
        loss, y, indices = self.infer(
            imgs,
            contrasts,
            target,
            self.vqvae,
            self.transformer
        )

        psnr = PSNR(max_val=1.0)(y, target)
        ssim = SSIM(spatial_dims=self.hparams.get('spatial_dims', 1),
                    data_range=1.0)(y, target)
        m_ssim = mSSIM(spatial_dims=self.hparams.get('spatial_dims', 1),
                       data_range=1.0)(y, target)
        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)
        assert isinstance(m_ssim, torch.Tensor)

        self.log('test/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test/m_ssim', m_ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.transformer.parameters(), lr=0e-4)

    # -------------------------------------------------------------------
    # private methods
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
