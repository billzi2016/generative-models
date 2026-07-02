"""
latent GAN 采样入口。

读取 gan/runs/latent_gan/best.pt 中的 EMA generator，生成 latent，并通过 VAE decoder
输出图片。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_VAE_DIR, load_best_vae, save_latent_grid, save_latent_images, select_device
from gan.models import LatentGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 latent GAN 生成图片")
    parser.add_argument("--checkpoint", default="gan/runs/latent_gan/best.pt")
    parser.add_argument("--vae-dir", default=str(DEFAULT_VAE_DIR))
    parser.add_argument("--output", default="gan/runs/latent_gan/samples.png")
    parser.add_argument("--output-dir", default=None, help="如果设置，则保存单张图片序列")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"), help="单张图片输出格式")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    noise_dim = int(config["noise_dim"])

    generator = LatentGenerator(noise_dim=noise_dim).to(device)
    generator.load_state_dict(checkpoint["ema_generator"])
    generator.eval()
    vae = load_best_vae(args.vae_dir, device=device)

    torch.manual_seed(args.seed)
    noise = torch.randn(args.num_images, noise_dim, device=device)
    latents = generator(noise)
    if args.output_dir:
        save_latent_images(vae, latents, args.output_dir, image_format=args.image_format)
        print(f"已保存样例目录: {args.output_dir}")
    else:
        save_latent_grid(vae, latents, args.output, nrow=4)
        print(f"已保存样例: {args.output}")


if __name__ == "__main__":
    main()
