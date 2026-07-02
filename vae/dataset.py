"""
VAE 数据集读取模块。

本文件负责把已经解压好的动漫头像图片目录转换成 PyTorch Dataset。
它位于整体训练流程的第一步：原始图片 -> VAE encoder -> latent。

设计原则：
- 默认匹配当前仓库中的 dataset/raw/fullMin256。
- 递归扫描图片文件，适配 DAF 这种按子目录分桶存放图片的结构。
- 在训练阶段按图像方向处理：竖图/正方形裁最上方正方形，普通横图直接 bicubic resize。
- 超宽横图在扫描阶段过滤为 bad，避免异常构图进入训练。
- 归一化到 [-1, 1]，方便 VAE decoder 使用 Tanh 输出。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_LANDSCAPE_ASPECT_RATIO = 16 / 9

# DAF 数据中可能存在少量截断 JPEG。允许 PIL 读取轻微截断图片，避免训练中途崩溃。
ImageFile.LOAD_TRUNCATED_IMAGES = True


def prepare_image_geometry(image: Image.Image) -> Image.Image:
    """
    按方向处理图片几何形状。

    - 竖图/正方形：裁最上方正方形，适合头像/头部数据。
    - 普通横图：不裁剪，后续直接 bicubic resize 到 128x128。
    - 超宽横图：不应进入这里，扫描阶段会标记为 bad。
    """
    width, height = image.size
    if height >= width:
        return image.crop((0, 0, width, width))
    return image


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


def _write_if_changed(path: Path, lines: list[str]) -> None:
    """
    仅当内容变化时写文件，并使用临时文件 + rename，避免运行中断留下半截文件。
    """
    content = "\n".join(lines)
    if content:
        content += "\n"

    if path.exists() and path.read_text(encoding="utf-8") == content:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _inspect_image(image_path: Path) -> tuple[Path, bool]:
    """
    检查单张图片是否可用于训练。

    返回:
        (image_path, is_valid)
    """
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        if height <= 0 or width / height >= MAX_LANDSCAPE_ASPECT_RATIO:
            return image_path, False
        return image_path, True
    except Exception:
        return image_path, False


def scan_image_lists(
    data_dir: str | Path,
    valid_list_path: str | Path = "dataset/valid_images.txt",
    bad_list_path: str | Path = "dataset/bad_images.txt",
    scan_workers: int = 8,
) -> list[Path]:
    """
    每次启动时全量扫描图片，生成当前可用图片列表。

    行为：
    - 总是扫描当前数据目录，带进度条，确保数据集变动能被发现。
    - 默认使用 8 个线程扫描图片，避免 33 万张图单线程扫描过慢。
    - 只有 valid/bad 列表内容变化时才写文件，避免没变化时重复写 SSD。
    - 返回本次扫描得到的 valid paths，训练直接使用内存结果，不依赖旧文件。
    """
    image_paths = find_image_files(data_dir)
    if not image_paths:
        raise RuntimeError(f"没有在目录中找到图片: {data_dir}")

    valid_paths: list[Path] = []
    bad_paths: list[Path] = []

    worker_count = max(1, scan_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(_inspect_image, image_paths)
        for image_path, is_valid in tqdm(results, total=len(image_paths), desc=f"scan images ({worker_count} workers)"):
            if is_valid:
                valid_paths.append(image_path)
            else:
                bad_paths.append(image_path)

    valid_lines = [str(path) for path in valid_paths]
    bad_lines = [str(path) for path in bad_paths]
    _write_if_changed(Path(valid_list_path), valid_lines)
    _write_if_changed(Path(bad_list_path), bad_lines)

    print(f"图片扫描完成: valid={len(valid_paths)}, bad={len(bad_paths)}")
    return valid_paths


class ImageFolderRecursiveDataset(Dataset):
    """
    递归图片数据集。

    输入是任意包含图片的目录，输出是训练 VAE 所需的图像 tensor。
    默认输出范围为 [-1, 1]，形状为 [3, image_size, image_size]。
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 128,
        valid_list_path: str | Path = "dataset/valid_images.txt",
        bad_list_path: str | Path = "dataset/bad_images.txt",
        scan_workers: int = 8,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.image_paths = scan_image_lists(self.data_dir, valid_list_path, bad_list_path, scan_workers=scan_workers)
        if not self.image_paths:
            raise RuntimeError(f"没有在目录中找到图片: {self.data_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Lambda(prepare_image_geometry),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
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
