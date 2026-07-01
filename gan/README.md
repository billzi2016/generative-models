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

默认读取：

```text
dataset/latents/vae_daf_128_best.h5
vae/runs/vae_daf/best_diffusers
```

## 训练

```bash
python gan/train.py \
  --latents-h5 dataset/latents/vae_daf_128_best.h5 \
  --output-dir gan/runs/latent_gan \
  --batch-size 128 \
  --epochs 100
```

## 采样

```bash
python gan/sample.py \
  --checkpoint gan/runs/latent_gan/best.pt \
  --vae-dir vae/runs/vae_daf/best_diffusers \
  --output gan/runs/latent_gan/samples.png
```

采样默认使用 checkpoint 中的 EMA generator。
