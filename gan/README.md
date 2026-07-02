# gan

This directory implements a `latent GAN` baseline.

It is an experimental baseline, not the main line. The main line should prioritize `ddpm/`, `dit/`, and `flow_matching/`.
The purpose of GAN here is to provide a faster-sampling comparison point, not to guarantee better stability than diffusion or flow methods.

Current setup:

- the generator outputs `[B, 4, 16, 16]` VAE scaled latents
- the discriminator works directly in latent space
- hinge loss is used
- the discriminator uses spectral normalization
- training maintains an EMA generator, and sampling uses EMA weights by default

## Required Inputs

```bash
python vae/cache_latents.py
```

The two VAE routes correspond to two sets of inputs:

```text
Route A: dataset/latents/vae_daf_128_finetune.h5
Route A: vae/runs/vae_daf_finetune/best_diffusers
Route A 10%: dataset/latents/vae_daf_128_finetune_10pct.h5
Route A 10%: vae/runs/vae_daf_finetune_10pct/best_diffusers

Route B: dataset/latents/vae_daf_128_from_scratch.h5
Route B: vae/runs/vae_daf_from_scratch/best_diffusers
Route B 10%: dataset/latents/vae_daf_128_from_scratch_10pct.h5
Route B 10%: vae/runs/vae_daf_from_scratch_10pct/best_diffusers
```

## Route A: Fine-Tune From A Pretrained VAE

```bash
# Train latent GAN on Route A latents.
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune.h5 \
  --output-dir gan/runs/latent_gan_finetune \
  --batch-size 128 \
  --epochs 100 \
  --patience 8 \
  --min-delta 1e-4

# Generate 128 jpg images with the Route A VAE decoder.
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_finetune/best.pt \
  --vae-dir vae/runs/vae_daf_finetune/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_finetune/samples_jpg \
  --image-format jpg
```

## Route A 10%: Fine-Tune From The 10% Quick VAE

```bash
# Train latent GAN on Route A 10% latents.
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune_10pct.h5 \
  --output-dir gan/runs/latent_gan_finetune_10pct \
  --batch-size 128 \
  --epochs 100 \
  --patience 3 \
  --min-delta 1e-4

# Generate 128 jpg images with the Route A 10% VAE decoder.
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_finetune_10pct/best.pt \
  --vae-dir vae/runs/vae_daf_finetune_10pct/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_finetune_10pct/samples_jpg \
  --image-format jpg
```

## Route B: Train The VAE From Scratch

```bash
# Train latent GAN on Route B latents.
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch.h5 \
  --output-dir gan/runs/latent_gan_from_scratch \
  --batch-size 128 \
  --epochs 100 \
  --patience 8 \
  --min-delta 1e-4

# Generate 128 jpg images with the Route B VAE decoder.
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_from_scratch/best.pt \
  --vae-dir vae/runs/vae_daf_from_scratch/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_from_scratch/samples_jpg \
  --image-format jpg
```

## Route B 10%: From The 10% Quick Scratch VAE

```bash
# Train latent GAN on Route B 10% latents.
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch_10pct.h5 \
  --output-dir gan/runs/latent_gan_from_scratch_10pct \
  --batch-size 128 \
  --epochs 100 \
  --patience 3 \
  --min-delta 1e-4

# Generate 128 jpg images with the Route B 10% VAE decoder.
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_from_scratch_10pct/best.pt \
  --vae-dir vae/runs/vae_daf_from_scratch_10pct/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_from_scratch_10pct/samples_jpg \
  --image-format jpg
```

Sampling uses the EMA generator from the checkpoint by default.

GAN early stopping uses `val_fake_score`, not a plain MSE metric: full-data routes use `--patience 8 --min-delta 1e-4`, while 10% quick-VAE routes use `--patience 3 --min-delta 1e-4`.
