"""模型前向 shape dry test。"""

from __future__ import annotations

import unittest

import torch

from ddpm.train import build_model as build_ddpm
from dit.train import build_model as build_dit
from flow_matching.train import build_model as build_flow
from gan.models import LatentDiscriminator, LatentGenerator


class ModelShapeTest(unittest.TestCase):
    def test_ddpm_unet_shape(self) -> None:
        model = build_ddpm()
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        y = model(x, t).sample
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))

    def test_dit_shape(self) -> None:
        model = build_dit()
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        class_labels = torch.zeros(2, dtype=torch.long)
        y = model(x, timestep=t, class_labels=class_labels).sample
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))

    def test_flow_unet_shape(self) -> None:
        model = build_flow()
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        y = model(x, t).sample
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))

    def test_latent_gan_shapes(self) -> None:
        generator = LatentGenerator(noise_dim=256)
        discriminator = LatentDiscriminator()
        noise = torch.randn(2, 256)
        fake = generator(noise)
        score = discriminator(fake)
        self.assertEqual(tuple(fake.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(score.shape), (2,))


if __name__ == "__main__":
    unittest.main()
