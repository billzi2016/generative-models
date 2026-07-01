# VAE 数据处理与保存策略

## 目标

本文件记录当前 VAE 阶段的数据处理、训练/验证、checkpoint 保存和 latent 缓存策略。

核心原则：

- VAE 是后续所有 latent 生成方法的统一接口。
- 默认使用 Diffusers `AutoencoderKL`，不维护手写 VAE。
- VAE 训练过程要能选出可靠的 `best` 权重。
- 保存策略必须控制 SSD 占用，不能无限堆 checkpoint。
- VAE 最终确定后，后续方法应读取一次性缓存好的 HDF5 latent，而不是重复跑 encoder。

## 数据来源

当前数据目录：

```text
dataset/raw/fullMin256
```

当前 VAE 输入分辨率固定为：

```text
128 x 128
```

VAE latent 形状固定为：

```text
[batch, 4, 16, 16]
```

## VAE 训练/验证划分

当前 `vae/train_vae.py` 使用标准 train / val 划分：

```text
train split: 用于反向传播和更新 VAE 参数
val split: 只计算 loss，不更新参数
```

默认验证集比例：

```text
val_ratio = 0.05
```

这样做的原因：

- 如果只是直接使用预训练 VAE，不需要训练/验证。
- 如果要在 DAF 数据上微调 VAE，需要 val loss 防止把预训练 VAE 调坏。
- `best_diffusers/` 根据 val loss 选择，而不是根据 train loss 选择。

注意：

- train/val 只用于 VAE 微调阶段。
- 后续缓存 latent 时，应该使用 `best_diffusers/` encode 全量数据，而不是只 encode train split。
- 后续 `ddpm/`、`dit/`、`flow_matching/`、`gan/` 可以在 HDF5 latent 上重新划分自己的 train/val。

## VAE 权重保存策略

默认必须保存：

```text
vae/runs/vae_daf/best_diffusers/
```

这是后续所有方法默认读取的 VAE 权重目录。

默认不保存：

```text
best.pt
last.pt
```

原因是 `.pt` 训练状态会包含 optimizer / scheduler state，体积明显更大。除非需要恢复 VAE 微调，否则不应该默认保存。

如果需要保存训练状态，显式传入：

```bash
python vae/train_vae.py --save-training-state
```

如果需要保存最后一轮 Diffusers 权重，显式传入：

```bash
python vae/train_vae.py --save-last
```

## 周期 checkpoint

可以按 epoch 间隔保存少量中间权重：

```bash
python vae/train_vae.py \
  --checkpoint-every-epochs 5 \
  --max-epoch-checkpoints 10
```

这会产生类似：

```text
epoch_0005_diffusers/
epoch_0010_diffusers/
epoch_0015_diffusers/
```

并自动只保留最新的 `max_epoch_checkpoints` 个周期 checkpoint。
默认保留数量为 `10`。

设计目的：

- 保留少量回退点。
- 避免每轮都保存导致 SSD 被 checkpoint 占满。
- 不影响 `best_diffusers/` 的保存。

## reconstruction 样例图

每次 val loss 创新低时，会保存一张重建对比图：

```text
reconstruction_epoch_XXX.png
```

这些图片用于肉眼检查 VAE 是否重建正常。它们体积很小，但后续也应该被 `.gitignore` 忽略。

## latent 缓存策略

VAE best 确定后，应缓存全量 latent，供后续方法复用：

```text
dataset/raw/fullMin256
  -> vae/runs/vae_daf/best_diffusers
  -> dataset/latents/vae_daf_128_best.h5
```

建议 HDF5 内容：

```text
latents: float16, shape [N, 4, 16, 16]
paths: 原图相对路径或字符串路径
```

建议 HDF5 属性：

```text
vae_path = vae/runs/vae_daf/best_diffusers
image_size = 128
latent_shape = [4, 16, 16]
scaling_factor = 0.18215
dtype = float16
posterior = mode
compression = gzip
compression_opts = 1
```

### h5py 保存示例

推荐用单个 HDF5 文件保存全量 latent，不要每张图一个文件。

```python
from pathlib import Path

import h5py
import numpy as np

output_path = Path("dataset/latents/vae_daf_128_best.h5")
output_path.parent.mkdir(parents=True, exist_ok=True)

num_images = 331844
latent_shape = (4, 16, 16)

with h5py.File(output_path, "w") as h5:
    latents_ds = h5.create_dataset(
        "latents",
        shape=(num_images, *latent_shape),
        dtype="float16",
        chunks=(256, *latent_shape),
        compression="gzip",
        compression_opts=1,
    )

    paths_ds = h5.create_dataset(
        "paths",
        shape=(num_images,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        compression="gzip",
        compression_opts=1,
    )

    h5.attrs["vae_path"] = "vae/runs/vae_daf/best_diffusers"
    h5.attrs["image_size"] = 128
    h5.attrs["latent_shape"] = np.array(latent_shape, dtype=np.int64)
    h5.attrs["scaling_factor"] = 0.18215
    h5.attrs["dtype"] = "float16"
    h5.attrs["posterior"] = "mode"
    h5.attrs["compression"] = "gzip"
    h5.attrs["compression_opts"] = 1

    # 实际缓存脚本中，每个 batch 生成一段 latent 后写入对应切片。
    # latents_np: [batch, 4, 16, 16], float16
    # paths: list[str]
    start = 0
    end = start + len(paths)
    latents_ds[start:end] = latents_np
    paths_ds[start:end] = paths
```

关键参数：

- `compression="gzip"`：启用 gzip 压缩。
- `compression_opts=1`：使用较轻的压缩级别，减少 CPU 压缩开销。
- `chunks=(256, 4, 16, 16)`：按 batch 附近大小分块，便于后续随机 batch 读取。
- `dtype="float16"`：latent 训练通常够用，体积比 float32 减半。

### h5py 读取示例

后续 `ddpm/`、`dit/`、`flow_matching/`、`gan/` 可以直接从 HDF5 读取 latent：

```python
import h5py
import torch
from torch.utils.data import Dataset


class LatentH5Dataset(Dataset):
    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self.h5 = None

        # 先打开一次只读元信息，避免在主进程长期持有文件句柄。
        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["latents"].shape[0]
            self.latent_shape = tuple(h5["latents"].shape[1:])

    def _file(self):
        # DataLoader 多 worker 时，每个 worker 应该懒加载自己的 h5 文件句柄。
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        h5 = self._file()
        latent = h5["latents"][index]
        return torch.from_numpy(latent).float()
```

训练脚本拿到的 batch 形状就是：

```text
[batch, 4, 16, 16]
```

缓存时使用：

```text
posterior.mode()
```

而不是：

```text
posterior.sample()
```

原因：

- `mode()` 是确定性的，便于复现。
- 后续所有生成方法看到的是同一份 latent 数据。
- 避免每次缓存因 posterior 采样不同而改变训练数据。

保存前应乘以 Diffusers scaling factor：

```python
latent = vae.encode(image).latent_dist.mode()
latent = latent * vae.config.scaling_factor
```

后续解码时再除回去：

```python
image = vae.decode(latent / vae.config.scaling_factor).sample
```

## SSD 占用预估

当前数据约 `331844` 张图。

### VAE checkpoint

默认只强制保存：

```text
best_diffusers/
config.json
reconstruction_epoch_*.png
```

其中 `best_diffusers/` 通常是几百 MB 量级，取决于保存格式和模型权重精度。

如果开启周期 checkpoint：

```bash
--checkpoint-every-epochs 5 --max-epoch-checkpoints 10
```

最多额外保留：

```text
10 个 epoch_XXXX_diffusers/
```

旧的周期 checkpoint 会自动删除，因此不会无限增长。

大致空间量级：

```text
best_diffusers/        约 300-400 MB
每个 epoch checkpoint 约 300-400 MB
10 个周期 checkpoint  约 3-4 GB
```

默认不保存 `.pt` optimizer / scheduler 状态，因为 AdamW optimizer state 通常会显著增加体积。只有需要恢复 VAE 微调时，才显式开启：

```bash
--save-training-state
```

### latent HDF5

单个 latent 大小：

```text
4 * 16 * 16 * float16 = 2048 bytes
```

全量 latent 原始大小约：

```text
331844 * 2048 bytes ≈ 680 MB
```

使用 HDF5 `gzip compression_opts=1` 后，实际大小取决于 latent 可压缩性，通常仍应控制在可接受范围内。

不建议：

- 把每张 latent 存成独立 `.pt` 文件。
- 每个方法各自重复 encode 一份 latent。
- 默认保存每个 epoch 的完整 VAE 权重。

## 后续方法读取约定

后续方法默认读取：

```text
dataset/latents/vae_daf_128_best.h5
```

包括：

- `ddpm/`
- `dit/`
- `flow_matching/`
- `gan/`

这些方法训练时不再默认读取原始图片，也不再默认调用 VAE encoder。

采样阶段仍需要 VAE decoder：

```text
generated latent -> VAE decoder -> image
```
