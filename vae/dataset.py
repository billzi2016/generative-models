"""
VAE 数据集读取模块。

本文件负责把已经解压好的动漫头像图片目录转换成 PyTorch Dataset。
它位于整体训练流程的第一步：原始图片 -> VAE encoder -> latent。

设计原则：
- 默认匹配当前仓库中的 dataset/raw/fullMin256。
- 递归扫描图片文件，适配 DAF 这种按子目录分桶存放图片的结构。
- 在训练阶段统一 resize 到指定分辨率，并归一化到 [-1, 1]，方便 VAE decoder 使用 Tanh 输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_image_files(root: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    """
    递归查找目录下的图片文件。

    参数：
        root: 已解压的数据目录，可以是具体图片目录，也可以是更上层目录。
        extensions: 允许的图片扩展名集合。

    返回：
        排序后的图片路径列表。排序是为了让同一份数据在不同机器上顺序尽量稳定。
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"数据路径不是目录: {root_path}")

    normalized_extensions = {ext.lower() for ext in extensions}
    image_paths = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    image_paths.sort()
    return image_paths


class ImageFolderRecursiveDataset(Dataset):
    """
    递归图片数据集。

    输入是任意包含图片的目录，输出是训练 VAE 所需的图像 tensor。
    默认输出范围为 [-1, 1]，形状为 [3, image_size, image_size]。
    """

    def __init__(self, data_dir: str | Path, image_size: int = 128) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.image_paths = find_image_files(self.data_dir)
        if not self.image_paths:
            raise RuntimeError(f"没有在目录中找到图片: {self.data_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        # 统一转成 RGB，避免 PNG 透明通道或灰度图破坏 batch shape。
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor
