from typing import Dict

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image


class Image2DWriter(pl.Callback):
    def __init__(self, log_dir, log_interval=100):
        super().__init__()
        self.log_dir = log_dir
        self.log_interval = log_interval

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            assert isinstance(outputs, Dict)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    grid = make_grid(
                        value, nrow=8, normalize=True, scale_each=True)
                    save_image(
                        grid, f"{self.log_dir}/val_{batch_idx}_{key}.png")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            assert isinstance(outputs, Dict)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    grid = make_grid(
                        value, nrow=8, normalize=True, scale_each=True)
                    save_image(
                        grid, f"{self.log_dir}/test_{batch_idx}_{key}.png")


class Image3DWriter(pl.Callback):
    def __init__(self, log_dir, log_interval=100):
        super().__init__()
        self.log_dir = log_dir
        self.log_interval = log_interval

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            ...

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            ...
