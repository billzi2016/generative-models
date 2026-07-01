"""
VAE latent 插值 GIF 生成脚本。

本文件用于验证和展示 VAE 的连续 latent 空间：
1. 读取两张输入图片。
2. 使用 Diffusers AutoencoderKL 编码成 SD 风格 latent。
3. 在 latent 空间做线性插值。
4. 使用 VAE decoder 解码每个插值 latent。
5. 保存为平滑过渡 GIF。

该过程不需要 DDPM / diffusion，因为目标是两张已知图片之间的平滑过渡，
不是从噪声分布采样生成新图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import DEFAULT_PRETRAINED_VAE, load_autoencoder_kl


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 VAE latent 插值生成变脸 GIF")
    parser.add_argument("--image-a", required=True, help="起始图片路径")
    parser.add_argument("--image-b", required=True, help="结束图片路径")
    parser.add_argument("--output", default="vae/runs/interpolation.gif", help="输出 GIF 路径")
    parser.add_argument("--vae", default=DEFAULT_PRETRAINED_VAE, help="Diffusers VAE 权重名或本地目录")
    parser.add_argument("--image-size", type=int, default=128, help="输入图片统一缩放尺寸，当前主线固定 128")
    parser.add_argument("--frames", type=int, default=32, help="GIF 帧数")
    parser.add_argument("--duration", type=int, default=80, help="每帧持续时间，单位毫秒")
    parser.add_argument("--loop", type=int, default=0, help="GIF 循环次数，0 表示无限循环")
    parser.add_argument("--use-posterior-sample", action="store_true", help="使用 posterior sample；默认使用 mode，过渡更稳定")
    return parser.parse_args()


def select_device() -> torch.device:
    """优先选择 CUDA，其次 Apple Silicon MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_tensor(path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    """
    加载单张图片并转换为 VAE 输入张量。

    输出形状为 [1, 3, image_size, image_size]，数值范围为 [-1, 1]。
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def encode_image(vae, image: torch.Tensor, use_posterior_sample: bool) -> torch.Tensor:
    """
    把图片编码成供后续插值使用的 scaled latent。

    Diffusers 约定：给 diffusion 用的 latent 需要乘以 scaling_factor。
    插值也放在这个同一尺度上，后续解码时再除回去。
    """
    posterior = vae.encode(image).latent_dist
    latent = posterior.sample() if use_posterior_sample else posterior.mode()
    return latent * vae.config.scaling_factor


@torch.no_grad()
def decode_latent(vae, scaled_latent: torch.Tensor) -> Image.Image:
    """
    把 scaled latent 解码成 PIL 图片。

    AutoencoderKL decoder 接收未缩放 latent，因此这里先除以 scaling_factor。
    """
    latent = scaled_latent / vae.config.scaling_factor
    image = vae.decode(latent).sample
    image = (image.clamp(-1, 1) + 1) / 2
    image = image.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(image)
    return image


def interpolate_latents(latent_a: torch.Tensor, latent_b: torch.Tensor, frames: int) -> list[torch.Tensor]:
    """在线性 latent 空间中生成插值序列。"""
    if frames < 2:
        raise ValueError("frames 必须大于等于 2")

    latents = []
    for index in range(frames):
        t = index / (frames - 1)
        latents.append((1 - t) * latent_a + t * latent_b)
    return latents


def main() -> None:
    """主流程：编码两张图，插值，解码，保存 GIF。"""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"使用设备: {device}")

    vae = load_autoencoder_kl(args.vae).to(device)
    vae.eval()
    print(
        "VAE latent 配置:",
        f"latent_channels={vae.config.latent_channels},",
        f"scaling_factor={vae.config.scaling_factor}",
    )

    image_a = load_image_tensor(args.image_a, args.image_size, device)
    image_b = load_image_tensor(args.image_b, args.image_size, device)

    latent_a = encode_image(vae, image_a, args.use_posterior_sample)
    latent_b = encode_image(vae, image_b, args.use_posterior_sample)
    print(f"latent shape: {tuple(latent_a.shape)}")

    frames = [decode_latent(vae, latent) for latent in interpolate_latents(latent_a, latent_b, args.frames)]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=args.loop,
    )
    print(f"已保存 GIF: {output_path}")


if __name__ == "__main__":
    main()
