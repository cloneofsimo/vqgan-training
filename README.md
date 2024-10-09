# VAE Trainer

The famous VAE of latent diffusion models, such as stable diffusion, FLUX, SORA, etc. How are they trained? This is my attempt to write distributed VAE trainer.

## Details

This project implements a distributed VAE trainer using PyTorch's DistributedDataParallel (DDP). The VAE architecture is based on the one used in latent diffusion models, with some modifications for improved training stability and performance.

## Key Features

1. **Distributed Training**: Utilizes PyTorch's DDP for efficient multi-GPU training.
2. **GAN Loss**: Optional GAN loss for improved image quality.
3. **Perceptual Loss**: Uses LPIPS for perceptual similarity.
4. **Gradient Normalization**: Custom gradient normalization for stable training.
5. **WebDataset Support**: Efficient data loading using WebDataset.

- `ae.py`: Contains the VAE architecture implementation.

- `vae_trainer.py`: Main training script with DDP setup.

- `utils.py`: Utility functions and classes, including LPIPS and PatchDiscriminator.

## Usage

To start training, use the following command:

```bash
torchrun --nproc_per_node=8 vae_trainer.py
```

This will initiate training on 8 GPUs. Adjust the number based on your available hardware.

## Configuration

The trainer supports various configuration options through command-line arguments. Some key parameters include:

- `--learning_rate_vae`: Learning rate for the VAE.
- `--vae_ch`: Base channel size for the VAE.
- `--vae_ch_mult`: Channel multipliers for the VAE.
- `--do_ganloss`: Flag to enable GAN loss.

For a full list of options, refer to the `train_ddp` function in `vae_trainer.py`.

## Loss Functions

The trainer uses a combination of losses for effective VAE training:

1. Reconstruction Loss: L1 loss between downsampled original and reconstructed images.
2. KL Divergence Loss: Modified KL loss with soft large penalty.
3. Perceptual Loss: LPIPS-based perceptual similarity.
4. GAN Loss (optional): Adversarial loss for improved image quality.


---

Above readme was written by claude.
