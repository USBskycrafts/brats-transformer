import random
import typing as t
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from nibabel import nifti1


class BraTS2021Dataset:
    def __init__(
        self,
        roots: str,
        modalites: t.Tuple[str],
        slice_range: t.List[int],
    ):

        self.roots = roots.split(';')
        self.modalites = modalites
        self.slice_range = slice_range

        if not all(Path(root).exists() for root in self.roots):
            raise FileNotFoundError(f"Dataset in {self.roots} does not exist.")
        self.slice_range = sorted(slice_range)  # Ensure range is sorted

        self.samples = []
        for root in self.roots:
            for sample in Path(root).iterdir():
                if sample.is_dir():
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples) * (self.slice_range[1] - self.slice_range[0])

    def __getitem__(self, idx: int):
        sample_idx = idx // (self.slice_range[1] - self.slice_range[0])
        slice_idx = idx % (self.slice_range[1] - self.slice_range[0])
        slice_idx = self.slice_range[0] + slice_idx
        sample = self.samples[sample_idx]
        modalities = []
        if getattr(self, 'normalize_params', None) is None:
            self.normalize_params = {}
        if self.normalize_params.get(sample_idx, None) is None:
            self.normalize_params[sample_idx] = {}

        for mod in self.modalites:
            mod_path = sample / f"{sample.stem}-{mod}.nii.gz"
            if not mod_path.exists():
                raise FileNotFoundError(
                    f"Modality {mod} for sample {sample} does not exist.")

            img = nifti1.load(mod_path)
            if self.normalize_params[sample_idx].get(mod, None) is None:
                data = img.get_fdata(dtype=np.float32)
                vmax = np.max(data)
                vmin = np.min(data)
                self.normalize_params[sample_idx][mod] = (vmin, vmax)

            img_slice = img.dataobj[:, :, slice_idx]
            vmin, vmax = self.normalize_params[sample_idx][mod]
            # normalize to [-1, 1]
            img_slice = (img_slice - vmin) / (vmax - vmin) * 2 - 1
            img_slice = np.clip(img_slice, -1, 1)
            img_slice = img_slice.astype(np.float32)

            modalities.append(img_slice)
        modalities = np.stack(modalities, axis=-1)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(192),
        ])

        idx_chosen = torch.randperm(len(self.modalites))
        input_modalities = idx_chosen[:(len(self.modalites) - 1)]
        target_modalities = idx_chosen[(len(self.modalites) - 1):]

        source = modalities[:, :, input_modalities]
        target = modalities[:, :, target_modalities]

        mask_length = random.randint(1, len(self.modalites) - 1)
        mask = [1] * mask_length \
            + [0] * (len(self.modalites) - 1 - mask_length)

        input_modalities = torch.tensor(mask) * (input_modalities + 1)
        target_modalities = target_modalities + 1

        source = transform(source)
        target = transform(target)
        return (
            source,
            input_modalities,
            target,
            target_modalities,
        )
