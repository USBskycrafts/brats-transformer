import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.vqvae import VQVAE
from transformer.transformer import ContrastGenerationTransformer
from einops import rearrange


class MultiContrastGenerationInferer(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                imgs,
                contrasts,
                target,
                vqvae: VQVAE,
                transformer: ContrastGenerationTransformer
                ):
        features = []
        with torch.no_grad():
            for img in imgs:
                features.append(vqvae.encode_stage_2_inputs(img))
            indices = vqvae.index_quantize(target)
            indices = indices.flatten(1)
        # y is a logits tensor
        y = transformer(features, contrasts)
        loss = self.loss(y.transpose(1, -1), indices)
        return loss, y, indices

    @torch.no_grad()
    def sample(self,
               imgs,
               contrasts,
               vqvae: VQVAE,
               transformer: ContrastGenerationTransformer,
               tau=1.0
               ):
        features = []
        for img in imgs:
            features.append(vqvae.encode_stage_2_inputs(img))
        # y is a logits tensor
        spatial_shape = features[0].shape[2:]
        y = transformer(features, contrasts)
        # using gumbel softmax to sample
        indices = torch.argmax(F.gumbel_softmax(y, tau=tau, hard=True), dim=2)
        if len(spatial_shape) == 2:
            h, w = spatial_shape
            indices = rearrange(indices, 'b (h w) -> b h w', h=h, w=w)
        elif len(spatial_shape) == 3:
            d, h, w = spatial_shape
            indices = rearrange(indices, 'b (d h w) -> b d h w', d=d, h=h, w=w)
        else:
            raise ValueError(
                f"Unsupported spatial shape: {spatial_shape}, expected 2D or 3D tensor.")
        target = vqvae.decode_samples(indices)
        return target
