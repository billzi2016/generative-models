"""Generate a 10-column x 6-row MNIST Flow Matching trajectory GIF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import MnistFlowUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MNIST Flow Matching GIF")
    parser.add_argument("--checkpoint", default="vis-flow-matching/runs/mnist_flow/best.pt")
    parser.add_argument("--output", default="vis-flow-matching/runs/mnist_flow/mnist_flow_10x6.gif")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=None, help="默认从 checkpoint config 读取")
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensors_to_grid_frame(images: torch.Tensor, rows: int, cols: int, scale: int) -> Image.Image:
    """Convert [rows*cols, 1, 28, 28] tensor in roughly [-1, 1] to a PIL grid."""
    images = images.detach().cpu().clamp(-1, 1)
    images = ((images + 1) * 127.5).to(torch.uint8)
    cells = images[:, 0].reshape(rows, cols, 28, 28)

    grid = Image.new("L", (cols * 28, rows * 28), color=0)
    for row in range(rows):
        for col in range(cols):
            cell = Image.fromarray(cells[row, col].numpy(), mode="L")
            grid.paste(cell, (col * 28, row * 28))
    if scale != 1:
        grid = grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST)
    return grid


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.cols != 10:
        raise ValueError("--cols 必须为 10，因为列对应数字 0..9")

    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    base_channels = args.base_channels or int(config.get("base_channels", 32))

    model = MnistFlowUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    torch.manual_seed(args.seed)
    sample_count = args.rows * args.cols
    x = torch.randn(sample_count, 1, 28, 28, device=device)
    labels = torch.arange(10, device=device).repeat(args.rows)

    frames: list[Image.Image] = [tensors_to_grid_frame(x, args.rows, args.cols, args.scale)]
    dt = 1.0 / args.steps
    for step in range(args.steps):
        t_value = torch.full((sample_count,), step / args.steps, device=device)
        velocity = model(x, t_value, labels)
        x = x + dt * velocity
        frames.append(tensors_to_grid_frame(x, args.rows, args.cols, args.scale))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / args.fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"已保存 GIF: {output}")


if __name__ == "__main__":
    main()
