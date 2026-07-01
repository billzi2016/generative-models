# dit

本目录实现 `latent DiT`。

DiT 和 DDPM 使用同样的 latent diffusion 训练目标，但 denoiser backbone 使用
Diffusers `DiTTransformer2DModel`，而不是 U-Net。

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
python dit/train.py \
  --latents-h5 dataset/latents/vae_daf_128_best.h5 \
  --output-dir dit/runs/latent_dit \
  --batch-size 128 \
  --epochs 100
```

默认 latent shape：

```text
[B, 4, 16, 16]
```

## 采样

```bash
python dit/sample.py \
  --model-dir dit/runs/latent_dit/best_model \
  --vae-dir vae/runs/vae_daf/best_diffusers \
  --output dit/runs/latent_dit/samples.png
```

采样使用 `DDIMScheduler`，默认 `50` 步。
