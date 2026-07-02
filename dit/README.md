# dit

本目录实现 `latent DiT`。

DiT 和 DDPM 使用同样的 latent diffusion 训练目标，但 denoiser backbone 使用
Diffusers `DiTTransformer2DModel`，而不是 U-Net。

## 依赖输入

```bash
python vae/cache_latents.py
```

两条 VAE 路线对应两套输入：

```text
路线 A：dataset/latents/vae_daf_128_finetune.h5
路线 A：vae/runs/vae_daf_finetune/best_diffusers

路线 B：dataset/latents/vae_daf_128_from_scratch.h5
路线 B：vae/runs/vae_daf_from_scratch/best_diffusers
```

## 路线 A：基于预训练 VAE 微调

```bash
# 训练 DiT，读取路线 A 的 latent 缓存。
python dit/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune.h5 \
  --output-dir dit/runs/latent_dit_finetune \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 A 的 VAE decoder。
python dit/sample.py \
  --model-dir dit/runs/latent_dit_finetune/best_model \
  --vae-dir vae/runs/vae_daf_finetune/best_diffusers \
  --num-images 128 \
  --output-dir dit/runs/latent_dit_finetune/samples_jpg \
  --image-format jpg
```

## 路线 B：从零训练 VAE

```bash
# 训练 DiT，读取路线 B 的 latent 缓存。
python dit/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch.h5 \
  --output-dir dit/runs/latent_dit_from_scratch \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 B 的 VAE decoder。
python dit/sample.py \
  --model-dir dit/runs/latent_dit_from_scratch/best_model \
  --vae-dir vae/runs/vae_daf_from_scratch/best_diffusers \
  --num-images 128 \
  --output-dir dit/runs/latent_dit_from_scratch/samples_jpg \
  --image-format jpg
```

默认 latent shape：

```text
[B, 4, 16, 16]
```

采样使用 `DDIMScheduler`，默认 `50` 步。
