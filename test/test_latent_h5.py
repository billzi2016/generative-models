"""HDF5 latent 读写 dry test。"""

from __future__ import annotations

from pathlib import Path
import unittest

import h5py
import numpy as np

from common.latent import LatentH5Dataset


class LatentH5Test(unittest.TestCase):
    def test_latent_h5_dataset_reads_float_tensor(self) -> None:
        tmp_dir = Path("test/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        h5_path = tmp_dir / "latents_test.h5"

        latents = np.random.randn(8, 4, 16, 16).astype("float16")
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset(
                "latents",
                data=latents,
                chunks=(4, 4, 16, 16),
                compression="gzip",
                compression_opts=1,
            )
            h5.attrs["scaling_factor"] = 0.18215

        dataset = LatentH5Dataset(h5_path)
        sample = dataset[0]

        self.assertEqual(len(dataset), 8)
        self.assertEqual(tuple(sample.shape), (4, 16, 16))
        self.assertEqual(str(sample.dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()
