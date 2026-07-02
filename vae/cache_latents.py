"""
缓存 VAE latent 到单个 HDF5 文件。

本脚本在 VAE best 权重确定后运行一次，把 dataset/raw/fullMin256 中的全量图片
编码成 scaled latent，并保存为 dataset/latents/vae_daf_128_best.h5。

后续 ddpm / dit / flow_matching / gan 都直接读取该 HDF5 文件，不在训练时重复运行
VAE encoder，从而提升训练速度并避免每个方法各自生成一份 latent 缓存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_LATENTS_H5, DEFAULT_VAE_DIR, load_best_vae, select_device
from vae.dataset import ImageFolderRecursiveDataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="缓存 VAE scaled latent 到 HDF5")
    parser.add_argument("--data-dir", default="dataset/raw/fullMin256", help="原始图片目录")
    parser.add_argument("--vae-dir", default=str(DEFAULT_VAE_DIR), help="VAE best_diffusers 权重目录")
    parser.add_argument("--output", default=str(DEFAULT_LATENTS_H5), help="输出 HDF5 路径")
    parser.add_argument("--image-size", type=int, default=128, help="VAE 输入尺寸")
    parser.add_argument("--batch-size", type=int, default=128, help="编码 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数量")
    parser.add_argument("--compression-opts", type=int, default=1, help="gzip 压缩等级，默认 1")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """主流程：读取图片，编码 latent，写入 HDF5。"""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"使用设备: {device}")

    dataset = ImageFolderRecursiveDataset(
        args.data_dir,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    vae = load_best_vae(args.vae_dir, device=device)
    latent_shape = (4, args.image_size // 8, args.image_size // 8)
    num_images = len(dataset)

    print(f"图片数量: {num_images}")
    print(f"latent shape: {latent_shape}")
    print(f"输出文件: {output_path}")

    with h5py.File(output_path, "w") as h5:
        latents_ds = h5.create_dataset(
            "latents",
            shape=(num_images, *latent_shape),
            dtype="float16",
            chunks=(min(args.batch_size, 256), *latent_shape),
            compression="gzip",
            compression_opts=args.compression_opts,
        )
        paths_ds = h5.create_dataset(
            "paths",
            shape=(num_images,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=args.compression_opts,
        )

        h5.attrs["vae_path"] = str(args.vae_dir)
        h5.attrs["image_size"] = args.image_size
        h5.attrs["latent_shape"] = np.array(latent_shape, dtype=np.int64)
        h5.attrs["scaling_factor"] = float(vae.config.scaling_factor)
        h5.attrs["dtype"] = "float16"
        h5.attrs["posterior"] = "mode"
        h5.attrs["compression"] = "gzip"
        h5.attrs["compression_opts"] = args.compression_opts

        offset = 0
        for images in tqdm(dataloader, desc="cache latents"):
            images = images.to(device)
            posterior = vae.encode(images).latent_dist
            latents = posterior.mode() * vae.config.scaling_factor
            latents_np = latents.detach().cpu().to(torch.float16).numpy()

            batch_size = latents_np.shape[0]
            start = offset
            end = offset + batch_size
            latents_ds[start:end] = latents_np
            paths_ds[start:end] = [str(path) for path in dataset.image_paths[start:end]]
            offset = end

    print(f"完成 latent 缓存: {output_path}")


if __name__ == "__main__":
    main()
