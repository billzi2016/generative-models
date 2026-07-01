"""语法级 dry test。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


class SyntaxTest(unittest.TestCase):
    def test_python_files_parse(self) -> None:
        files = [
            "common/latent.py",
            "common/training.py",
            "vae/cache_latents.py",
            "vae/train_vae.py",
            "ddpm/train.py",
            "ddpm/sample.py",
            "dit/train.py",
            "dit/sample.py",
            "flow_matching/train.py",
            "flow_matching/sample.py",
            "gan/models.py",
            "gan/train.py",
            "gan/sample.py",
        ]
        for file in files:
            with self.subTest(file=file):
                ast.parse(Path(file).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
