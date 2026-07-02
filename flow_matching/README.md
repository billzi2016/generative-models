# flow_matching

本目录实现 `latent Flow Matching / Rectified Flow`。

训练目标是学习 latent 空间中的连续时间速度场：

```text
z_t = (1 - t) z_0 + t z_1
v_theta(z_t, t) ≈ z_1 - z_0
```

其中 `z_1` 来自 VAE latent HDF5，`z_0` 是高斯噪声。

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
# 训练 Flow Matching，读取路线 A 的 latent 缓存。
python flow_matching/train.py \
  --latents-h5 dataset/latents/vae_daf_128_finetune.h5 \
  --output-dir flow_matching/runs/latent_fm_finetune \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 A 的 VAE decoder。
python flow_matching/sample.py \
  --model-dir flow_matching/runs/latent_fm_finetune/best_model \
  --vae-dir vae/runs/vae_daf_finetune/best_diffusers \
  --num-images 128 \
  --output-dir flow_matching/runs/latent_fm_finetune/samples_jpg \
  --image-format jpg
```

## 路线 B：从零训练 VAE

```bash
# 训练 Flow Matching，读取路线 B 的 latent 缓存。
python flow_matching/train.py \
  --latents-h5 dataset/latents/vae_daf_128_from_scratch.h5 \
  --output-dir flow_matching/runs/latent_fm_from_scratch \
  --batch-size 128 \
  --epochs 100

# 生成 128 张 jpg 图片，使用路线 B 的 VAE decoder。
python flow_matching/sample.py \
  --model-dir flow_matching/runs/latent_fm_from_scratch/best_model \
  --vae-dir vae/runs/vae_daf_from_scratch/best_diffusers \
  --num-images 128 \
  --output-dir flow_matching/runs/latent_fm_from_scratch/samples_jpg \
  --image-format jpg
```

采样使用 Euler ODE 积分，默认 `50` 步。
