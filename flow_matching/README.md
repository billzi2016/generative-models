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

默认读取：

```text
dataset/latents/vae_daf_128_best.h5
vae/runs/vae_daf/best_diffusers
```

## 训练

```bash
python flow_matching/train.py \
  --latents-h5 dataset/latents/vae_daf_128_best.h5 \
  --output-dir flow_matching/runs/latent_fm \
  --batch-size 128 \
  --epochs 100
```

## 采样

```bash
python flow_matching/sample.py \
  --model-dir flow_matching/runs/latent_fm/best_model \
  --vae-dir vae/runs/vae_daf/best_diffusers \
  --output flow_matching/runs/latent_fm/samples.png
```

采样使用 Euler ODE 积分，默认 `50` 步。
