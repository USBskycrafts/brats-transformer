# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act
# from monai.networks.layers.vector_quantizer import EMAQuantizer, VectorQuantizer
from autoencoder.quantizer import FiniteScalarQuantizer
from monai.utils.misc import ensure_tuple_rep

__all__ = ["VQVAE"]


class LearnablePositionEncoding(nn.Module):
    def __init__(self, spatial_dims: int, channels: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, channels, *([1]*spatial_dims)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class Attention(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, heads: int = 8):
        super().__init__()
        assert in_channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.head_dim = in_channels // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=3 * in_channels,
            kernel_size=1,
            conv_only=True
        )
        self.pos_enc = LearnablePositionEncoding(spatial_dims, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h d) ... -> b h (...) d', h=self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h ... d -> b (h d) (...)', h=self.heads)
        return self.gamma * out.view_as(x) + x


class VQVAEResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network as originally used in Morphology-preserving
    Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf).

    The original implementation that can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L150.

    Args:
        spatial_dims: number of spatial spatial_dims of the input data.
        in_channels: number of input channels.
        num_res_channels: number of channels in the residual layers.
        act: activation type and arguments. Defaults to RELU.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_channels: int,
        act: tuple | str | None = Act.RELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_res_channels = num_res_channels
        self.act = act
        self.dropout = dropout
        self.bias = bias

        self.conv1 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.num_res_channels,
            adn_ordering="NDA",
            act=self.act,
            norm=("GROUP", {"num_groups": 32}),
            dropout=self.dropout,
            bias=self.bias,
        )

        self.conv2 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_res_channels,
            out_channels=self.in_channels,
            bias=self.bias,
            conv_only=True,
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.conv2(self.conv1(x)), True)


class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of channels in the latent space (embedding_dim).
        channels: sequence containing the number of channels at each level of the encoder.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        downsample_parameters: Sequence[Tuple[int, int, int, int]],
        dropout: float,
        act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.act = act

        blocks: list[nn.Module] = []

        for i in range(len(self.channels)):
            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.in_channels if i == 0 else self.channels[i - 1],
                    out_channels=self.channels[i],
                    strides=self.downsample_parameters[i][0],
                    kernel_size=self.downsample_parameters[i][1],
                    adn_ordering="NDA",
                    act=self.act,
                    norm=("GROUP", {"num_groups": 32}),
                    dropout=None if i == 0 else self.dropout,
                    dropout_dim=1,
                    dilation=self.downsample_parameters[i][2],
                    padding=self.downsample_parameters[i][3],
                )
            )

            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        in_channels=self.channels[i],
                        num_res_channels=self.num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )
        blocks.append(
            Attention(
                spatial_dims=self.spatial_dims,
                in_channels=self.channels[len(self.channels) - 1],
            )
        )
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.channels[len(self.channels) - 1],
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of channels in the latent space (embedding_dim).
        out_channels: number of output channels.
        channels: sequence containing the number of channels at each level of the decoder.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
        output_act: activation type and arguments for the output.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        upsample_parameters: Sequence[Tuple[int, int, int, int, int]],
        dropout: float,
        act: tuple | str | None,
        output_act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.upsample_parameters = upsample_parameters
        self.dropout = dropout
        self.act = act
        self.output_act = output_act

        reversed_num_channels = list(reversed(self.channels))

        blocks: list[nn.Module] = []
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=reversed_num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        blocks.append(
            Attention(
                spatial_dims=self.spatial_dims,
                in_channels=reversed_num_channels[0],
            )
        )
        reversed_num_res_channels = list(reversed(self.num_res_channels))
        for i in range(len(self.channels)):
            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        in_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )

            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=reversed_num_channels[i],
                    out_channels=self.out_channels if i == len(
                        self.channels) - 1 else reversed_num_channels[i + 1],
                    strides=self.upsample_parameters[i][0],
                    kernel_size=self.upsample_parameters[i][1],
                    adn_ordering="NDA",
                    act=self.act,
                    dropout=self.dropout if i != len(
                        self.channels) - 1 else None,
                    norm=("GROUP", {"num_groups": 32}),
                    dilation=self.upsample_parameters[i][2],
                    conv_only=i == len(self.channels) - 1,
                    is_transposed=True if i != len(
                        self.channels) - 1 else False,
                    padding=self.upsample_parameters[i][3],
                    output_padding=self.upsample_parameters[i][4],
                )
            )

        if self.output_act:
            blocks.append(Act[self.output_act]())

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    """
    Vector-Quantised Variational Autoencoder (VQ-VAE) used in Morphology-preserving Autoregressive 3D Generative
    Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf)

    The original implementation can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L163/

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of output channels.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        num_res_layers: number of sequential residual layers at each level.
        channels: number of channels at each level.
        num_res_channels: number of channels in the residual layers at each level.
        num_embeddings: VectorQuantization number of atomic elements in the codebook.
        embedding_dim: VectorQuantization number of channels of the input and atomic elements.
        commitment_cost: VectorQuantization commitment_cost.
        decay: VectorQuantization decay.
        epsilon: VectorQuantization epsilon.
        act: activation type and arguments.
        dropout: dropout ratio.
        output_act: activation type and arguments for the output.
        ddp_sync: whether to synchronize the codebook across processes.
        use_checkpointing if True, use activation checkpointing to save memory.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (96, 96, 192),
        num_res_layers: int = 3,
        num_res_channels: Sequence[int] | int = (96, 96, 192),
        downsample_parameters: Sequence[Tuple[int, int, int, int]] | Tuple[int, int, int, int] = (
            (2, 4, 1, 1),
            (2, 4, 1, 1),
            (2, 4, 1, 1),
        ),
        upsample_parameters: Sequence[Tuple[int, int, int, int, int]] | Tuple[int, int, int, int, int] = (
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
        ),
        embedding_dim: int = 64,
        dropout: float = 0.0,
        act: tuple | str | None = Act.RELU,
        output_act: tuple | str | None = None,
        use_checkpointing: bool = False,
        levels: Sequence[int] = [8, 8, 8, 6, 5]
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.use_checkpointing = use_checkpointing

        num_res_channels = list(ensure_tuple_rep(
            num_res_channels, len(channels)))
        if len(num_res_channels) != len(channels):
            raise ValueError(
                "`num_res_channels` should be a single integer or a tuple of integers with the same length as "
                "`num_channls`."
            )
        # Ensure upsample_parameters is a sequence of tuples
        if isinstance(upsample_parameters, tuple) and all(isinstance(v, int) for v in upsample_parameters):
            upsample_parameters_tuple = [upsample_parameters] * len(channels)
        elif isinstance(upsample_parameters, Sequence):
            upsample_parameters_tuple = upsample_parameters
        else:
            raise TypeError(
                "upsample_parameters must be a tuple or sequence of tuples")

        # Ensure downsample_parameters is a sequence of tuples
        if isinstance(downsample_parameters, tuple) and all(isinstance(v, int) for v in downsample_parameters):
            downsample_parameters_tuple = [
                downsample_parameters] * len(channels)
        elif isinstance(downsample_parameters, Sequence):
            downsample_parameters_tuple = downsample_parameters
        else:
            raise TypeError(
                "downsample_parameters must be a tuple or sequence of tuples")

        if not all(all(isinstance(value, int) for value in sub_item) for sub_item in downsample_parameters_tuple):
            raise ValueError(
                "`downsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        # check if downsample_parameters is a tuple of ints or a tuple of tuples of ints
        if not all(all(isinstance(value, int) for value in sub_item) for sub_item in upsample_parameters_tuple):
            raise ValueError(
                "`upsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        for parameter in downsample_parameters_tuple:
            if len(parameter) != 4:
                raise ValueError(
                    "`downsample_parameters` should be a tuple of tuples with 4 integers.")

        for parameter in upsample_parameters_tuple:
            if len(parameter) != 5:
                raise ValueError(
                    "`upsample_parameters` should be a tuple of tuples with 5 integers.")

        if len(downsample_parameters_tuple) != len(channels):
            raise ValueError(
                "`downsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        if len(upsample_parameters_tuple) != len(channels):
            raise ValueError(
                "`upsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embedding_dim,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters_tuple,
            dropout=dropout,
            act=act,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            in_channels=embedding_dim,
            out_channels=out_channels,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters_tuple,
            dropout=dropout,
            act=act,
            output_act=output_act,
        )

        assert len(levels) == embedding_dim
        self.quantizer = FiniteScalarQuantizer(
            spatial_dims=spatial_dims,
            levels=levels,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor
        if self.use_checkpointing:
            output = torch.utils.checkpoint.checkpoint(
                self.encoder, images, use_reentrant=False)
        else:
            output = self.encoder(images)
        return output

    def quantize(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_loss, x = self.quantizer(encodings)
        return x, x_loss

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        if self.use_checkpointing:
            output = torch.utils.checkpoint.checkpoint(
                self.decoder, quantizations, use_reentrant=False)
        else:
            output = self.decoder(quantizations)
        return output

    def index_quantize(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(self.encode(images=images))

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.embed(embedding_indices))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantizations, quantization_losses = self.quantize(self.encode(images))
        reconstruction = self.decode(quantizations)

        return reconstruction, quantization_losses

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        e, _ = self.quantize(z)
        return e

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.quantize(z)
        image = self.decode(e)
        return image
