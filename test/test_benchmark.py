import unittest

import torch

from src.data import NamedImageDataset
from src.benchmark import benchmark_aggregate_metrics, benchmark_single_metrics


class TestMetric(unittest.TestCase):
    def test_benchmark(self):
        ex_names = ["a", "b", "c"]
        B = len(ex_names)
        H = 32
        W = 32
        ex_imgs = torch.arange(B * 3 * H * W).reshape(B, 3, H, W) / (B * 3 * H * W)
        ex_imgs_2 = ex_imgs + torch.randn(ex_imgs.shape, generator=torch.manual_seed(0))

        ex_imgs = torch.clamp(ex_imgs, 0, 1)
        ex_imgs_2 = torch.clamp(ex_imgs_2, 0, 1)

        preds = NamedImageDataset(ex_names, ex_imgs_2)
        gts = NamedImageDataset(ex_names, ex_imgs)

        sing_metrics = benchmark_single_metrics(preds, gts)
        agg_metrics = benchmark_aggregate_metrics(preds, gts)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
