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

默认读取：

```text
dataset/latents/vae_daf_128_best.h5
vae/runs/vae_daf/best_diffusers
```

## 训练

```bash
python ddpm/train.py \
  --latents-h5 dataset/latents/vae_daf_128_best.h5 \
  --output-dir ddpm/runs/latent_ddpm \
  --batch-size 128 \
  --epochs 100
```

输出目录：

```text
ddpm/runs/latent_ddpm/
```

默认保存：

- `best_model/`
- `last_model/`
- `scheduler/`
- 少量 `epoch_XXXX_model/`，数量由 `--max-epoch-checkpoints` 控制

## 采样

```bash
python ddpm/sample.py \
  --model-dir ddpm/runs/latent_ddpm/best_model \
  --vae-dir vae/runs/vae_daf/best_diffusers \
  --output ddpm/runs/latent_ddpm/samples.png
```

采样使用 `DDIMScheduler`，默认 `50` 步。
