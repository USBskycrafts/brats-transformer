# BraTS 2023 Medical Image Generation Project

## Overview

This project implements a medical image generation system using VQ-VAE and Transformer architectures for BraTS 2023 brain tumor MRI data. The system supports multi-dataset training (GLI, MEN, MET) and produces high-quality synthetic medical images.

## Key Features

- 🧠 Multi-modal MRI generation (T1, T1ce, T2, FLAIR)
- ⚡ Two-stage training (VQGAN + MaskGit)
- 📊 Comprehensive metrics tracking (PSNR, SSIM, LPIPS)
- 🖼️ Built-in visualization tools

## Dataset Configuration

```
/data/
└── BraTS2023/
    ├── GLI/
    │   ├── TrainingData/
    │   └── ValidationData/
    ├── MEN/
    │   ├── TrainingData/
    │   └── ValidationData/
    └── MET/
        ├── TrainingData/
        └── ValidationData/
```

## Complete Configuration Examples

### Stage 1 (VQGAN) Configuration - FSQ-F16-D5

```yaml
dataset:
  train:
    target: dataset.brats_multiple_2d_pretrained.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  val:
    target: dataset.brats_multiple_2d_pretrained.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  test:
    target: dataset.brats_multiple_2d_pretrained.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  batch_size: 256

model:
  target: autoencoder.vqvae.VQGAN
  params:
    spatial_dims: 2
    in_channels: 1
    out_channels: 1
    channels: [32, 32, 64, 128, 256]
    num_res_layers: 2
    num_res_channels: [32, 32, 64, 128, 256]
    downsample_parameters:
      [[1, 7, 1, 3], [2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1]]
    upsample_parameters:
      [
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [1, 7, 1, 3, 0],
      ]
    embedding_dim: 5
    levels: [8, 8, 8, 6, 5]
    act: memswish
    adv_weight: 0.01
    perceptual_weight: 0.1
    lr: 1.0e-4

trainer:
  max_epochs: 1000
  callbacks:
    - target: pytorch_lightning.callbacks.ModelCheckpoint
      params:
        dirpath: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5/
        filename: best-checkpoint-epoch{epoch:02d}-psnr{val/psnr:.2f}-ssim{val/ssim:.2f}
        monitor: val/ploss
        mode: min
        save_top_k: 5
        auto_insert_metric_name: False
    - target: callbacks.image_writer.Image2DWriter
      params:
        log_interval: 5
        log_dir: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5/
    - target: pytorch_lightning.callbacks.ModelSummary
      params:
        max_depth: 2

  logger:
    target: pytorch_lightning.loggers.TensorBoardLogger
    params:
      save_dir: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5/
      name: logging
```

### Stage 2 (MaskGit) Configuration - Any

```yaml
dataset:
  train:
    target: dataset.brats_multiple_2d.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  val:
    target: dataset.brats_multiple_2d.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  test:
    target: dataset.brats_multiple_2d.BraTS2021Dataset
    params:
      roots: [YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData/;[YOUR_DATA_PATH]/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/
      modalites: ["t1n", "t1c", "t2w", "t2f"]
      slice_range: [27, 127]
  batch_size: 128

model:
  target: transformer.maskgit_generator.ContrastMaskGiT
  params:
    spatial_dims: 2
    img_size: [12, 12]
    stage_one_ckpt: [YOUR_MODEL_PATH]/vqvae-multiple_dataset-fsq-f16-d5/best-checkpoint-epoch***-psnr****-ssim****.ckpt
    in_channels: 1
    out_channels: 1
    channels: [32, 32, 64, 128, 256]
    num_res_layers: 2
    num_res_channels: [32, 32, 64, 128, 256]
    downsample_parameters:
      [[1, 7, 1, 3], [2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1]]
    upsample_parameters:
      [
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [2, 4, 1, 1, 0],
        [1, 7, 1, 3, 0],
      ]
    levels: [8, 8, 8, 6, 5]
    embedding_dim: 5
    act: memswish
    num_contrast: 4
    hidden_size: 768
    mlp_dim: 3072
    num_layers: 12
    num_heads: 12
    lr: 1.0e-4

trainer:
  max_epochs: 1000
  accumulate_grad_batches: 4
  callbacks:
    - target: pytorch_lightning.callbacks.ModelCheckpoint
      params:
        dirpath: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5-stage2-any/
        filename: best-checkpoint-epoch{epoch:02d}-psnr{val/psnr:.2f}-ssim{val/ssim:.2f}
        monitor: val/ploss
        mode: min
        save_top_k: 5
        auto_insert_metric_name: False
    - target: callbacks.image_writer.Image2DWriter
      params:
        log_interval: 5
        log_dir: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5-stage2-any/
    - target: pytorch_lightning.callbacks.ModelSummary
      params:
        max_depth: 2

  logger:
    target: pytorch_lightning.loggers.TensorBoardLogger
    params:
      save_dir: [YOUR_RESULT_PATH]/vqvae-multiple_dataset-fsq-f16-d5-stage2-any/
      name: logging
```

### Configuration Notes

- `[YOUR_DATA_PATH]`: Replace with your actual data directory path
- `[YOUR_MODEL_PATH]`: Replace with your pretrained model path
- `[YOUR_RESULT_PATH]`: Replace with your result saving directory path

## Getting Started

### Installation

```bash
conda env create -f environment.yaml
conda activate maskgit
```

### Training Commands

```bash
# Stage 1: VQGAN Training
python main.py --config config/local_brats2023_synth.yaml --gpus 0,1 [--resume [YOUR_MODEL_PATH]]

# or just a single GPU
python main.py --config config/local_brats2023_synth.yaml --gpus 0 [--resume [YOUR_MODEL_PATH]]

# Stage 2: MaskGit Training
python main.py --config config/local_brats2023_synth-fsq-f16-d5-stage2-any.yaml --gpus 0,1 [--resume [YOUR_MODEL_PATH]]
```

### Testing Only Commands

```bash
# Stage 1 for example
python main.py --config config/local_brats2023_synth.yaml --gpus 0,1 --test-only True

```

### Monitoring

```bash
tensorboard --logdir [YOUR_RESULT_PATH]/logging --port 6006
```

## Project Structure

```
├── autoencoder/      # VQ-VAE implementation
├── transformer/      # MaskGit implementation
├── config/           # Configuration files
├── dataset/          # Data loading
├── callbacks/        # Training callbacks
├── main.py           # Main entry
└── utils.py          # Utilities
```

## References

```bibtex


@article{chang2022maskgit,
  title={Masked Generative Image Transformer},
  author={Chang, Huiwen and Zhang, Han and Jiang, Lu and Carlos Niebles, Juan and Freeman, William T},
  journal={arXiv preprint arXiv:2202.04200},
  year={2022}
}

@article{mentzer2023finite,
  title={Finite Scalar Quantization: VQ-VAE Made Simple},
  author={Mentzer, Fabian and Tschannen, Michael and Teterwak, Peter and Saurous, Rif A and Koltun, Vladlen and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2309.15505},
  year={2023}
}

@article{monai2020,
  title={MONAI: Medical Open Network for AI},
  author={MONAI Consortium},
  journal={arXiv preprint arXiv:2111.06194},
  year={2021}
}
```
