"""
latent Flow Matching 采样入口。

从高斯噪声 latent 出发，用训练好的 velocity field 做 Euler ODE 积分：
dz/dt = v_theta(z, t)

积分完成后使用 VAE decoder 生成图片。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"urllib3 .* doesn't match a supported version!")

import torch
from diffusers import UNet2DModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_VAE_DIR, load_best_vae, save_latent_grid, save_latent_images, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 latent Flow Matching 生成图片")
    parser.add_argument("--model-dir", default="flow_matching/runs/latent_fm/best_model")
    parser.add_argument("--vae-dir", default=str(DEFAULT_VAE_DIR))
    parser.add_argument("--output", default="flow_matching/runs/latent_fm/samples.png")
    parser.add_argument("--output-dir", default=None, help="如果设置，则保存单张图片序列")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"), help="单张图片输出格式")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    model = UNet2DModel.from_pretrained(args.model_dir).to(device)
    model.eval()
    vae = load_best_vae(args.vae_dir, device=device)

    latents = torch.randn(args.num_images, 4, 16, 16, generator=generator, device=device)
    dt = 1.0 / args.num_steps

    for step in tqdm(range(args.num_steps), desc="sample flow"):
        t = torch.full((args.num_images,), int(step / args.num_steps * 999), device=device, dtype=torch.long)
        velocity = model(latents, t).sample
        latents = latents + velocity * dt

    if args.output_dir:
        save_latent_images(vae, latents, args.output_dir, image_format=args.image_format)
        print(f"已保存样例目录: {args.output_dir}")
    else:
        save_latent_grid(vae, latents, args.output, nrow=4)
        print(f"已保存样例: {args.output}")


if __name__ == "__main__":
    main()
