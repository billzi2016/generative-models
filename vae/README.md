# vae

This directory prepares the front-end `VAE` used by the current small `Stable Diffusion`-style pipeline.

The main line does not maintain a hand-written VAE. It uses Hugging Face Diffusers `AutoencoderKL` throughout.

## Current Goal

- take anime portrait images as input
- load pretrained `stabilityai/sd-vae-ft-mse` by default
- if adaptation to DAF is needed, fine-tune Diffusers `AutoencoderKL` instead of maintaining a custom VAE
- output Diffusers-native `encoder / decoder` checkpoints
- let the latent generators in `ddpm/`, `dit/`, and `flow_matching/` consume the encoder output and decode generated latents back into images with the decoder

## Latent Shape

The current VAE follows the Stable Diffusion spatial latent interface instead of compressing the whole image into one vector.

Default setting:

```text
128 x 128 x 3 -> 4 x 16 x 16
```

Meaning:

- spatial downsampling by `8`
- latent channel count `4`
- total element count goes from `49152` to `1024`, roughly `48x` compression

Later U-Net, DiT, and Flow Matching models all consume latents multiplied by `scaling_factor`:

```text
[batch, 4, 16, 16]
```

The default Diffusers `AutoencoderKL` `scaling_factor` is `0.18215`. Diffusion-style training uses:

```python
latents = vae.encode(images).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

Decoding uses:

```python
images = vae.decode(latents / vae.config.scaling_factor).sample
```

## Data Path

The current `DAF` data has been extracted under:

```text
dataset/raw/
```

The current image directory is:

```text
dataset/raw/fullMin256
```

The VAE training script uses this directory by default. Images are grouped in subdirectories, for example:

```text
dataset/raw/fullMin256/0192/79192.jpg
```

Image preprocessing policy:

```text
extra-wide landscape: width / height >= 16 / 9 -> bad
portrait / square: crop the top square region -> bicubic resize to 128x128
normal landscape: no crop -> bicubic resize to 128x128
finally normalize to [-1, 1]
```

This filters obvious non-portrait extra-wide images while avoiding direct geometric distortion on portrait images.

Dataset scanning policy:

- every training or latent-caching run rescans the image directory so dataset changes and interrupted runs do not leave stale lists behind
- the default is `--scan-workers 8` with a `tqdm` progress bar; these workers are used for concurrent PIL open / validation
- results are written to `dataset/valid_images.txt` and `dataset/bad_images.txt`, but files are only rewritten when contents change, to avoid unnecessary SSD writes
- training reads from `valid_images.txt`, so bad images do not enter the DataLoader and crash training halfway through

If the image directory changes later, pass it explicitly:

```bash
python vae/train_vae.py --data-dir your_actual_image_dir
```

## Download Pretrained Weights

If you follow the "start from someone else's trained VAE and fine-tune it" route, download the weights locally first:

```bash
python vae/download_pretrained.py \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir vae/pretrained/sd-vae-ft-mse
```

Notes:

- `stabilityai/sd-vae-ft-mse` is a Hugging Face model id, not a local path
- `vae/pretrained/sd-vae-ft-mse` is the downloaded local directory and is already ignored by `.gitignore`

## Training / Fine-Tuning Examples

Route A: fine-tune a pretrained VAE.

```bash
python vae/train_vae.py \
  --data-dir dataset/raw/fullMin256 \
  --pretrained-vae vae/pretrained/sd-vae-ft-mse \
  --output-dir vae/runs/vae_daf_finetune \
  --image-size 128 \
  --batch-size 64 \
  --epochs 50 \
  --patience 8 \
  --min-delta 1e-4 \
  --checkpoint-every-epochs 5 \
  --max-epoch-checkpoints 10
```

If you only want to validate the training loop, MPS, checkpoints, and reconstruction outputs first, run a fixed-random `10%` subset of valid images:

```bash
python vae/train_vae.py \
  --data-dir dataset/raw/fullMin256 \
  --pretrained-vae vae/pretrained/sd-vae-ft-mse \
  --output-dir vae/runs/vae_daf_finetune_10pct \
  --image-size 128 \
  --batch-size 64 \
  --epochs 50 \
  --dataset-fraction 0.1 \
  --seed 42 \
  --patience 3 \
  --min-delta 1e-4 \
  --checkpoint-every-epochs 5 \
  --max-epoch-checkpoints 10
```

Note: `--dataset-fraction 0.1` samples a fixed random `10%` subset from valid images after bad-image scanning completes, using `--seed 42`. The default is `1.0`, which means full-data training. Full-data commands use explicit early stopping `--patience 8 --min-delta 1e-4`; the 10% quick validation route uses `--patience 3 --min-delta 1e-4`.

Route B: initialize Diffusers `AutoencoderKL` from scratch and train your own VAE.

```bash
python vae/train_vae.py \
  --data-dir dataset/raw/fullMin256 \
  --init-from-scratch \
  --output-dir vae/runs/vae_daf_from_scratch \
  --image-size 128 \
  --batch-size 64 \
  --epochs 50 \
  --patience 8 \
  --min-delta 1e-4 \
  --checkpoint-every-epochs 5 \
  --max-epoch-checkpoints 10
```

The two routes write to different directories so they do not overwrite each other:

```text
Route A: vae/runs/vae_daf_finetune/best_diffusers
Route B: vae/runs/vae_daf_from_scratch/best_diffusers
```

## Output Files

If `--output-dir` is not passed explicitly, the default output directory is:

```text
vae/runs/vae_daf/
```

The current recommendation is to always pass `--output-dir` explicitly for both routes.

Main files:

- `best_diffusers/`: the Diffusers VAE with the best validation loss, preferred for later `AutoencoderKL.from_pretrained()`
- `last_diffusers/`: the last-epoch Diffusers VAE, only saved if `--save-last` is passed
- `best.pt`: optimizer, scheduler, metrics, and other training state, only saved if `--save-training-state` is passed
- `last.pt`: last-epoch training state, only saved if `--save-training-state` is passed
- `config.json`: training configuration for this run
- `reconstruction_epoch_*.png`: original vs reconstructed comparisons, used to inspect VAE quality

For more details on data treatment, train / validation behavior, checkpointing, and latent caching, see:

```text
vae/dataset_treatment.md
```

## Bad Image Handling

Training and latent caching both scan the current data directory automatically at startup and show a progress bar.

Results are written to:

```text
dataset/valid_images.txt
dataset/bad_images.txt
```

Behavior:

- every run rescans the current data directory, so dataset changes are picked up
- if the scan result is identical to the existing files, they are not rewritten, which avoids unnecessary SSD writes
- if a previous run was interrupted and files are missing or incomplete, the current scan rewrites complete files atomically
- the dataset reads from the current valid list, so bad images do not trigger crashes in the middle of training

## Latent Interpolation GIF

The VAE can directly create a smooth transition between two images without DDPM or diffusion.

![VAE latent interpolation](runs/interpolation_pretrained.gif)

Pipeline:

```text
image A -> VAE encoder -> latent A
image B -> VAE encoder -> latent B
interpolate between latent A and latent B
interpolated latent -> VAE decoder -> GIF frames
```

Example:

```bash
python vae/interpolate_gif.py \
  --image-a dataset/raw/fullMin256/0987/1223987.jpg \
  --image-b dataset/raw/fullMin256/0987/3174987.jpg \
  --output vae/runs/interpolation_pretrained.gif
```

If `--vae` is not passed, the script uses Hugging Face `stabilityai/sd-vae-ft-mse` by default. If you want to inspect interpolation from your DAF fine-tuned VAE, pass the local `best_diffusers` explicitly:

```bash
python vae/interpolate_gif.py \
  --image-a dataset/raw/fullMin256/0987/1223987.jpg \
  --image-b dataset/raw/fullMin256/0987/3174987.jpg \
  --vae vae/runs/vae_daf_finetune/best_diffusers \
  --output vae/runs/interpolation_finetune.gif
```

The script uses `MPS` by default when available, and the latent output shape should be:

```text
[1, 4, 16, 16]
```

This script only uses the VAE encoder / decoder for linear interpolation in latent space. It does not use DDPM, DiT, Flow Matching, or GAN.
