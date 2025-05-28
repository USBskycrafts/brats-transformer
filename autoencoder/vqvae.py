from ast import List
from typing import Iterable, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.metrics.regression import MultiScaleSSIMMetric as mSSIM
from monai.metrics.regression import PSNRMetric as PSNR
from monai.networks.layers.factories import Act
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator

from .modules import VQVAE


class VQGAN(pl.LightningModule):
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 channels: Sequence[int] = (32, 32, 64, 128, 256),
                 num_res_layers: int = 2,
                 num_res_channels: Sequence[int] | int = (
                     32, 32, 64, 128, 256),
                 downsample_parameters: Sequence[tuple] = (
                     (1, 7, 1, 3),
                     (2, 4, 1, 1),
                     (2, 4, 1, 1),
                     (2, 4, 1, 1),
                     (2, 4, 1, 1),
                 ),
                 upsample_parameters: Sequence[tuple] = (
                     (2, 4, 1, 1, 0),
                     (2, 4, 1, 1, 0),
                     (2, 4, 1, 1, 0),
                     (2, 4, 1, 1, 0),
                     (1, 7, 1, 3, 0),
                 ),
                 num_embeddings: int = 1024,
                 embedding_dim: int = 3,
                 act: tuple | str | None = "SWISH",
                 adv_weight: float = 0.01,
                 perceptual_weight: float = 0.1,
                 lr=1e-4
                 ):
        super().__init__()

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

        self.discriminator = PatchDiscriminator(
            channels=64,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_layers_d=3,
        )

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type='alex'
        )
        self.l1_loss = nn.L1Loss()
        self.adv_loss = PatchAdversarialLoss(
            criterion='least_squares'
        )

        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x):
        return self.vqvae(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        optimizers = self.optimizers()
        assert isinstance(optimizers, Iterable)
        vae_opt, d_opt = optimizers

        # train VQVAE
        vae_opt.zero_grad()
        x_recon, quant_loss = self.vqvae(x)
        logits_fakes = self.discriminator(
            x_recon)[-1]

        recon_loss = self.l1_loss(x, x_recon)
        percetual_loss = self.perceptual_loss(x, x_recon)
        adv_loss = self.adv_loss(logits_fakes,
                                 target_is_real=True,
                                 for_discriminator=False)
        loss = recon_loss + quant_loss \
            + self.hparams.get('adv_weight', 0.01) * adv_loss \
            + self.hparams.get('perceptual_weight', 0.1) * percetual_loss

        self.manual_backward(loss)
        vae_opt.step()

        # train discriminator
        d_opt.zero_grad()
        x_recon, *_ = self.vqvae(x)
        logits_reals = self.discriminator(x.detach())[-1]
        loss_d_real = self.adv_loss(
            logits_reals,
            target_is_real=True,
            for_discriminator=True
        )
        logits_fakes = self.discriminator(x_recon.detach())[-1]
        loss_d_fake = self.adv_loss(
            logits_fakes,
            target_is_real=False,
            for_discriminator=True
        )
        loss_d = (loss_d_real + loss_d_fake) / 2
        self.manual_backward(loss_d)
        d_opt.step()

        # logging
        self.log('train/recon_loss', recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train/quant_loss', quant_loss, on_step=True,
                 on_epoch=True, prog_bar=False)
        self.log('train/adv_loss', adv_loss, on_step=True,
                 on_epoch=True, prog_bar=False)
        self.log('train/p_loss', percetual_loss, on_step=True,
                 on_epoch=True, prog_bar=False)
        self.log('train/loss_fake', loss_d_fake, on_step=True,
                 on_epoch=True, prog_bar=False)
        self.log('train/loss_real', loss_d_real, on_step=True,
                 on_epoch=True, prog_bar=False)

        return {
            'input': x,
            'generated': x_recon,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_recon, _ = self.vqvae(x)

        x_recon = x_recon.clamp(-1, 1)
        psnr = PSNR(max_val=2)(x, x_recon)
        ssim = mSSIM(self.hparams.get('spatial_dims', 2),
                     data_range=2)(x, x_recon)
        perceptual_loss = self.perceptual_loss(x, x_recon)

        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)
        self.log('val/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('val/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('val/ploss', perceptual_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {
            'input': x,
            'generated': x_recon,
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_recon, _ = self.vqvae(x)

        x_recon = x_recon.clamp(-1, 1)
        psnr = PSNR(max_val=2)(x, x_recon)
        ssim = mSSIM(self.hparams.get('spatial_dims', 2),
                     data_range=2)(x, x_recon)
        perceptual_loss = self.perceptual_loss(x, x_recon)

        assert isinstance(psnr, torch.Tensor)
        assert isinstance(ssim, torch.Tensor)
        self.log('test/psnr', psnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('test/ssim', ssim.mean(), on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('test/ploss', perceptual_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        return {
            'input': x,
            'generated': x_recon,
        }

    def configure_optimizers(self):
        opt_vqvae = torch.optim.Adam(
            self.vqvae.parameters(), lr=self.hparams.get('lr', 1e-4))
        opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.get('lr', 1e-4))
        return [opt_vqvae, opt_discriminator], []

    # -----------------------------------------------------------------
    # private methods
    def _save_vqvae_params(self, path: str):
        torch.save(self.vqvae.state_dict(), path)
