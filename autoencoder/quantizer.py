from typing import List, Sequence

import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, spatial_dims: int, levels: List[int]):
        super().__init__()
        # 使用PyTorch张量替代NumPy数组
        self.register_buffer(
            '_levels_tensor', torch.tensor(levels, dtype=torch.int32))

        # 计算basis张量
        cumprod = torch.cumprod(self._levels_tensor[:-1], dim=0)
        basis = torch.cat([torch.ones(1, dtype=torch.int32), cumprod])
        self.register_buffer('_basis', basis)

        # 维度排列设置
        self.flatten_permutation = [0] + list(range(2, spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [
            0, spatial_dims + 1] + list(range(1, spatial_dims + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(self.flatten_permutation)
        quantized = self._round_ste(self._bound(x))

        # 使用PyTorch计算half_width
        half_width = self._levels_tensor // 2
        quantized_normalized = quantized / half_width

        return 0.0, quantized_normalized.permute(self.quantization_permutation)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, encodings = self(encodings)
        encodings = encodings.permute(self.flatten_permutation)
        assert encodings.shape[-1] == len(self._levels_tensor)
        zhat = self._scale_and_shift(encodings)
        return (zhat * self._basis).sum(dim=-1).long()

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        embedding_indices = embedding_indices[..., None]
        # 纯PyTorch实现的整数除法与取余
        codes_non_centered = torch.remainder(
            torch.div(embedding_indices, self._basis, rounding_mode='floor'),
            self._levels_tensor
        )
        return self._scale_and_shift_inverse(codes_non_centered).permute(self.quantization_permutation)

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels_tensor // 2
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_tensor // 2
        return (zhat - half_width) / half_width

    def _bound(self, z: torch.Tensor):
        eps = 1e-3
        half_l = (self._levels_tensor - 1) * (1 - eps) / 2
        offset = torch.where(
            self._levels_tensor % 2 == 1,
            torch.tensor(0.0, device=z.device),
            torch.tensor(0.5, device=z.device)
        )
        shift = torch.atan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        return (torch.round(z) - z).detach() + z
