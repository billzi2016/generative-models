# ddpm

本目录实现 `latent DDPM`。

它不是像素空间 DDPM，而是在 VAE 缓存好的 latent 上训练：

```text
[B, 4, 16, 16] -> DDPMScheduler 加噪 -> UNet2DModel 预测噪声
```

## 依赖输入

先完成：

```bash
python vae/cache_latents.py
```

两条 VAE 路线对应两套输入：

```text
路线 A：dataset/latents/vae_daf_128_finetune.h5
路线 A：vae/runs/vae_daf_finetune/best_diffusers
路线 A 10%：dataset/latents/vae_daf_128_finetune_10pct.h5
路线 A 10%：vae/runs/vae_daf_finetune_10pct/best_diffusers

路线 B：dataset/latents/vae_daf_128_from_scratch.h5
路线 B：vae/runs/vae_daf_from_scratch/best_diffusers
路线 B 10%：dataset/latents/vae_daf_128_from_scratch_10pct.h5
路线 B 10%：vae/runs/vae_daf_from_scratch_10pct/best_diffusers
```

## 路线 A：基于预训练 VAE 微调

```bash
# 训练 DDPM，读取路线 A 的 latent 缓存。
python ddpm/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune.h5 \
  --output-dir ddpm/runs/latent_ddpm_finetune \
  --batch-size 128 \
  --epochs 100 \
  --patience 8 \
  --min-delta 1e-4

# 生成 128 张 jpg 图片，使用路线 A 的 VAE decoder。
python ddpm/sample.py \
  --model-dir ddpm/runs/latent_ddpm_finetune/best_model \
  --vae-dir vae/runs/vae_daf_finetune/best_diffusers \
  --num-images 128 \
  --output-dir ddpm/runs/latent_ddpm_finetune/samples_jpg \
  --image-format jpg
```

## 路线 A 10%：基于 10% 快速 VAE 微调

```bash
# 训练 DDPM，读取路线 A 10% 快速 VAE 的 latent 缓存。
python ddpm/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune_10pct.h5 \
  --output-dir ddpm/runs/latent_ddpm_finetune_10pct \
  --batch-size 128 \
  --epochs 100 \
  --patience 3 \
  --min-delta 1e-4

# 生成 128 张 jpg 图片，使用路线 A 10% 快速 VAE decoder。
python ddpm/sample.py \
  --model-dir ddpm/runs/latent_ddpm_finetune_10pct/best_model \
  --vae-dir vae/runs/vae_daf_finetune_10pct/best_diffusers \
  --num-images 128 \
  --output-dir ddpm/runs/latent_ddpm_finetune_10pct/samples_jpg \
  --image-format jpg
```

## 路线 B：从零训练 VAE

```bash
# 训练 DDPM，读取路线 B 的 latent 缓存。
python ddpm/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch.h5 \
  --output-dir ddpm/runs/latent_ddpm_from_scratch \
  --batch-size 128 \
  --epochs 100 \
  --patience 8 \
  --min-delta 1e-4

# 生成 128 张 jpg 图片，使用路线 B 的 VAE decoder。
python ddpm/sample.py \
  --model-dir ddpm/runs/latent_ddpm_from_scratch/best_model \
  --vae-dir vae/runs/vae_daf_from_scratch/best_diffusers \
  --num-images 128 \
  --output-dir ddpm/runs/latent_ddpm_from_scratch/samples_jpg \
  --image-format jpg
```

## 路线 B 10%：基于 10% 快速从零 VAE

```bash
# 训练 DDPM，读取路线 B 10% 快速 VAE 的 latent 缓存。
python ddpm/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch_10pct.h5 \
  --output-dir ddpm/runs/latent_ddpm_from_scratch_10pct \
  --batch-size 128 \
  --epochs 100 \
  --patience 3 \
  --min-delta 1e-4

# 生成 128 张 jpg 图片，使用路线 B 10% 快速 VAE decoder。
python ddpm/sample.py \
  --model-dir ddpm/runs/latent_ddpm_from_scratch_10pct/best_model \
  --vae-dir vae/runs/vae_daf_from_scratch_10pct/best_diffusers \
  --num-images 128 \
  --output-dir ddpm/runs/latent_ddpm_from_scratch_10pct/samples_jpg \
  --image-format jpg
```

## 输出

默认保存：

- `best_model/`
- `last_model/`
- `scheduler/`
- 少量 `epoch_XXXX_model/`，数量由 `--max-epoch-checkpoints` 控制

采样使用 `DDIMScheduler`，默认 `50` 步。

Early stopping 按 `val_loss` 判断：全量路线使用 `--patience 8 --min-delta 1e-4`，10% 快速 VAE 路线使用 `--patience 3 --min-delta 1e-4`。
