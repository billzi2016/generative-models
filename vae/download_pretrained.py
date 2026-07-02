"""
下载预训练 VAE 权重到本地目录。

本脚本只做一件事：把 Hugging Face Diffusers 的 AutoencoderKL 权重下载下来，
并用 save_pretrained 保存成本地目录。后续微调时使用这个本地目录作为
--pretrained-vae 参数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusers import AutoencoderKL


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="下载 Diffusers 预训练 VAE 到本地")
    parser.add_argument("--model-id", default="stabilityai/sd-vae-ft-mse", help="Hugging Face VAE model id")
    parser.add_argument("--output-dir", default="vae/pretrained/sd-vae-ft-mse", help="本地保存目录")
    return parser.parse_args()


def main() -> None:
    """下载并保存 VAE 权重。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.model_id)
    vae.save_pretrained(output_dir, safe_serialization=True)
    print(f"已保存 VAE 到: {output_dir}")


if __name__ == "__main__":
    main()
