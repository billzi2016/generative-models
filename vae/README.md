# vae

本目录用于准备当前小型 `Stable Diffusion` 路线的前置 `VAE`。

当前主线不手写 VAE 网络，统一使用 Hugging Face Diffusers 的 `AutoencoderKL`。

## 当前目标

- 输入动漫头像图片。
- 默认加载预训练 `stabilityai/sd-vae-ft-mse`。
- 如确实要适配 DAF 数据，也用 Diffusers `AutoencoderKL` 微调，不再维护自定义 VAE。
- 输出 Diffusers 原生格式的 `encoder / decoder` checkpoint。
- 后续 `ddpm/`、`dit/`、`flow_matching/` 等目录中的 latent 生成模型将使用这里的 `encoder` 产生空间 latent map，并使用 `decoder` 将生成 latent 解码回图像。

## Latent 形状

当前 VAE 采用 Stable Diffusion 的空间 latent 接口，而不是把整张图压成单个向量。

默认设置：

```text
128 x 128 x 3 -> 4 x 16 x 16
```

含义：

- 空间尺寸下采样 `8` 倍。
- latent 通道数为 `4`。
- 总元素数从 `49152` 变成 `1024`，约 `48` 倍压缩。

后续 U-Net、DiT 或 Flow Matching 都统一吃乘过 `scaling_factor` 的 latent：

```text
[batch, 4, 16, 16]
```

Diffusers `AutoencoderKL` 的默认 `scaling_factor` 是 `0.18215`。后续 diffusion 训练使用：

```python
latents = vae.encode(images).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

解码时使用：

```python
images = vae.decode(latents / vae.config.scaling_factor).sample
```

## 数据路径

当前 `DAF` 数据已经解压在：

```text
dataset/raw/
```

当前实际图片目录为：

```text
dataset/raw/fullMin256
```

VAE 训练脚本默认直接使用这个目录。目录内图片按子目录分桶，例如：

```text
dataset/raw/fullMin256/0192/79192.jpg
```

图片预处理策略：

```text
超宽横图: width / height >= 16 / 9 -> bad
竖图/正方形: 裁剪最上方正方形区域 -> bicubic resize 到 128x128
普通横图: 不裁剪 -> bicubic resize 到 128x128
最后统一归一化到 [-1, 1]
```

这样可以过滤明显不适合头像训练的超宽横图，同时避免竖图被直接拉伸变形。

数据扫描策略：

- 每次训练或缓存 latent 前都会重新扫描图片，避免数据集变化或上次中断后沿用旧清单。
- 默认使用 `--scan-workers 8` 并显示 `tqdm` 进度条；这里是并发扫描 worker，用于加速 PIL 打开和图片校验。
- 扫描结果写入 `dataset/valid_images.txt` 和 `dataset/bad_images.txt`，但只有内容变化时才重写文件，避免每次启动都重复写 SSD。
- 训练集实际读取 `valid_images.txt`，坏图不会进入 DataLoader，因此不会在训练中途因为截断图片崩溃。

如果后续换了数据目录，训练时显式传入即可：

```bash
python vae/train_vae.py --data-dir 实际图片目录
```

## 下载预训练权重

如果走“基于别人训练好的 VAE 再微调”这条路线，先下载权重到本地：

```bash
python vae/download_pretrained.py \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir vae/pretrained/sd-vae-ft-mse
```

说明：

- `stabilityai/sd-vae-ft-mse` 是 Hugging Face model id，不是本地文件路径。
- `vae/pretrained/sd-vae-ft-mse` 是下载后的本地目录，已加入 `.gitignore`。

## 训练 / 微调示例

路线 A：基于预训练 VAE 微调。

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

如果只是先确认训练流程、MPS、checkpoint 和 reconstruction 输出是否正常，可以固定随机抽 `10%` valid 图片跑快速版：

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

说明：`--dataset-fraction 0.1` 会在完成坏图扫描后，从 valid 图片中按 `--seed 42` 固定随机抽样 `10%`。默认值是 `1.0`，也就是全量训练。全量训练命令显式使用 early stopping：`--patience 8 --min-delta 1e-4`；10% 快速验证显式使用 `--patience 3 --min-delta 1e-4`。

路线 B：从 Diffusers `AutoencoderKL` 架构随机初始化，训练自己的 VAE。

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

两条路线输出到不同目录，避免互相覆盖：

```text
路线 A：vae/runs/vae_daf_finetune/best_diffusers
路线 B：vae/runs/vae_daf_from_scratch/best_diffusers
```

## 输出文件

如果不显式传 `--output-dir`，默认输出目录：

```text
vae/runs/vae_daf/
```

当前建议两条路线都显式指定 `--output-dir`。

主要文件：

- `best_diffusers/`：验证集 loss 最优的 Diffusers VAE，后续优先用 `AutoencoderKL.from_pretrained()` 读取。
- `last_diffusers/`：最后一轮 Diffusers VAE，需要显式传入 `--save-last` 才保存。
- `best.pt`：优化器、scheduler、指标等训练状态，需要显式传入 `--save-training-state` 才保存。
- `last.pt`：最后一轮训练状态，需要显式传入 `--save-training-state` 才保存。
- `config.json`：本次训练配置。
- `reconstruction_epoch_*.png`：原图和重建图对比，用于检查 VAE 质量。

更完整的数据处理、train/val、checkpoint 和 latent 缓存策略见：

```text
vae/dataset_treatment.md
```

## 坏图处理

训练和缓存 latent 启动时都会自动扫描当前数据目录，并显示进度条。

扫描结果固定写到：

```text
dataset/valid_images.txt
dataset/bad_images.txt
```

行为：

- 每次启动都会重新扫描当前数据目录，数据集变动可以被发现。
- 如果扫描结果和已有文件完全一致，不会重写文件，避免无意义写 SSD。
- 如果上次运行中断导致文件缺失或不完整，本次扫描会重新原子写入完整文件。
- Dataset 使用本次扫描得到的 valid 列表，坏图不会在训练中途才触发崩溃。

## Latent 插值 GIF

VAE 可以直接做两张图之间的平滑过渡，不需要 DDPM / diffusion。

![VAE latent interpolation](runs/interpolation_pretrained.gif)

流程：

```text
image A -> VAE encoder -> latent A
image B -> VAE encoder -> latent B
latent A/B 插值
插值 latent -> VAE decoder -> GIF 帧
```

运行示例：

```bash
python vae/interpolate_gif.py \
  --image-a dataset/raw/fullMin256/0987/1223987.jpg \
  --image-b dataset/raw/fullMin256/0987/3174987.jpg \
  --output vae/runs/interpolation_pretrained.gif
```

上面这条命令不传 `--vae`，默认使用 Hugging Face 上的 `stabilityai/sd-vae-ft-mse`。如果要看自己在 DAF 上微调后的 VAE 插值，需要显式指定本地 `best_diffusers`：

```bash
python vae/interpolate_gif.py \
  --image-a dataset/raw/fullMin256/0987/1223987.jpg \
  --image-b dataset/raw/fullMin256/0987/3174987.jpg \
  --vae vae/runs/vae_daf_finetune/best_diffusers \
  --output vae/runs/interpolation_finetune.gif
```

默认会使用 `MPS`，输出 latent 形状应为：

```text
[1, 4, 16, 16]
```

这个脚本只用 VAE encoder / decoder 做 latent 空间线性插值，不用 DDPM、DiT、Flow Matching 或 GAN。
