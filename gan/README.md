# gan

本目录实现 `latent GAN` 生成基线。

它不是当前 Stable Diffusion 风格主线的优先方法，但可以作为采样速度快的横向对照。

当前配置：

- Generator 输出 `[B, 4, 16, 16]` VAE scaled latent。
- Discriminator 在 latent 空间判别真假。
- 使用 hinge loss。
- Discriminator 使用 spectral norm。
- 训练时维护 EMA generator，采样默认使用 EMA 权重。

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
# 训练 latent GAN，读取路线 A 的 latent 缓存。
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune.h5 \
  --output-dir gan/runs/latent_gan_finetune \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 A 的 VAE decoder。
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_finetune/best.pt \
  --vae-dir vae/runs/vae_daf_finetune/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_finetune/samples_jpg \
  --image-format jpg
```

## 路线 B：从零训练 VAE

```bash
# 训练 latent GAN，读取路线 B 的 latent 缓存。
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch.h5 \
  --output-dir gan/runs/latent_gan_from_scratch \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 B 的 VAE decoder。
python gan/sample.py \
  --checkpoint gan/runs/latent_gan_from_scratch/best.pt \
  --vae-dir vae/runs/vae_daf_from_scratch/best_diffusers \
  --num-images 128 \
  --output-dir gan/runs/latent_gan_from_scratch/samples_jpg \
  --image-format jpg
```

采样默认使用 checkpoint 中的 EMA generator。
