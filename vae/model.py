"""
Diffusers AutoencoderKL 封装。

本文件不再手写 VAE 结构，而是统一使用 Hugging Face Diffusers 的 AutoencoderKL。
这样后续 latent diffusion / stable_diffusion 代码可以直接沿用 Diffusers 生态的
encode、decode、save_pretrained、from_pretrained 和 scaling_factor 约定。
"""

from __future__ import annotations

from diffusers import AutoencoderKL


DEFAULT_PRETRAINED_VAE = "stabilityai/sd-vae-ft-mse"


def load_autoencoder_kl(
    pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_VAE,
    *,
    init_from_scratch: bool = False,
) -> AutoencoderKL:
    """
    加载或创建 AutoencoderKL。

    参数：
        pretrained_model_name_or_path: Diffusers VAE 权重名或本地目录。
        init_from_scratch: 为 True 时只使用 Diffusers 的成熟模型类，但从随机权重开始。

    返回：
        AutoencoderKL 实例。默认预训练 SD VAE 的 latent 约定是：
        [B, 3, 128, 128] -> [B, 4, 16, 16]，scaling_factor=0.18215。
    """
    if not init_from_scratch:
        return AutoencoderKL.from_pretrained(pretrained_model_name_or_path)

    # 备用方案：不用手写网络，只用 Diffusers 官方 AutoencoderKL 类从零训练。
    # block_out_channels 和 4 个 down/up block 对应 8 倍空间下采样。
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=4,
        sample_size=128,
        scaling_factor=0.18215,
        norm_num_groups=32,
    )
