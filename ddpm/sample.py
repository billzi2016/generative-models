"""
latent DDPM 采样入口。

读取 ddpm/runs/latent_ddpm/best_model，使用 DDIMScheduler 从噪声 latent 采样，
再通过 VAE best decoder 还原成图片。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_VAE_DIR, load_best_vae, save_latent_grid, save_latent_images, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 latent DDPM 生成图片")
    parser.add_argument("--model-dir", default="ddpm/runs/latent_ddpm/best_model")
    parser.add_argument("--vae-dir", default=str(DEFAULT_VAE_DIR))
    parser.add_argument("--output", default="ddpm/runs/latent_ddpm/samples.png")
    parser.add_argument("--output-dir", default=None, help="如果设置，则保存单张图片序列")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"), help="单张图片输出格式")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
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

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    latents = torch.randn(args.num_images, 4, 16, 16, generator=generator, device=device)
    for timestep in tqdm(scheduler.timesteps, desc="sample ddpm"):
        noise_pred = model(latents, timestep).sample
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    if args.output_dir:
        save_latent_images(vae, latents, args.output_dir, image_format=args.image_format)
        print(f"已保存样例目录: {args.output_dir}")
    else:
        save_latent_grid(vae, latents, args.output, nrow=4)
        print(f"已保存样例: {args.output}")


if __name__ == "__main__":
    main()
