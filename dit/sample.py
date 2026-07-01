"""
latent DiT 采样入口。

使用训练好的 DiTTransformer2DModel 预测噪声，DDIMScheduler 反推 latent，再用 VAE
decoder 还原图片。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DiTTransformer2DModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_VAE_DIR, load_best_vae, save_latent_grid, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 latent DiT 生成图片")
    parser.add_argument("--model-dir", default="dit/runs/latent_dit/best_model")
    parser.add_argument("--vae-dir", default=str(DEFAULT_VAE_DIR))
    parser.add_argument("--output", default="dit/runs/latent_dit/samples.png")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    model = DiTTransformer2DModel.from_pretrained(args.model_dir).to(device)
    model.eval()
    vae = load_best_vae(args.vae_dir, device=device)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    latents = torch.randn(args.num_images, 4, 16, 16, generator=generator, device=device)
    class_labels = torch.zeros(args.num_images, device=device, dtype=torch.long)
    for timestep in tqdm(scheduler.timesteps, desc="sample dit"):
        timesteps = torch.full((args.num_images,), int(timestep), device=device, dtype=torch.long)
        noise_pred = model(latents, timestep=timesteps, class_labels=class_labels).sample
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    save_latent_grid(vae, latents, args.output, nrow=4)
    print(f"已保存样例: {args.output}")


if __name__ == "__main__":
    main()
