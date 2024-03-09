import pathlib as pl
from pathlib import Path
from collections.abc import Iterable

import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import read_image, load_img_if_path


def benchmark_fid(fid_metric, pred: Tensor, gt: Tensor) -> Tensor:
    fid_metric.reset()
    fid_metric.update(pred, real=False)
    fid_metric.update(gt, real=True)
    fid = fid_metric.compute()
    return fid

default_metrics = None

fid_metric = FrechetInceptionDistance(feature=64)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")

default_funcs = {
    "psnr": psnr_metric.__call__,
    "ssim": ssim_metric.__call__,
    "lpips": lpips_metric.__call__,
    "fid": lambda pred, gt: benchmark_fid(fid_metric, pred, gt),
}


class Metrics:
    def __init__(self, metric_funcs = None) -> None:
        if metric_funcs is None:
            metric_funcs = default_funcs

        self.funcs = metric_funcs

    @property
    def default(self):
        return default_metrics

    @property
    def metric_names(self):
        return list(self.funcs.keys())

    def benchmark_metric(self, metric_name: str, pred: Tensor, gt: Tensor):
        if not metric_name in self.metric_names:
            raise ValueError(f"Metric {metric_name} not found in metrics {self.metric_names}")

        with torch.no_grad():
            return self.funcs[metric_name](pred, gt)

    def benchmark(self, pred: Tensor | Path | str, gt: Tensor | Path | str, metric_names: Iterable[str] = None):
        pred = load_img_if_path(pred)
        gt = load_img_if_path(gt)
            
        if metric_names is None:
            metric_names = self.metric_names

        elif isinstance(metric_names, str):
            metric_names = [metric_names]

        return {metric_name: self.benchmark_metric(metric_name, pred, gt) for metric_name in metric_names}

    def benchmark_dir_to_dir(
        self, pred_dir_path: Path, gt_dir_path: Path, metric_names: str | Iterable[str] = None
    ) -> dict[str, float]:
        img_name_to_pred_path = {}
        for path in pred_dir_path.glob("*.jpg"):
            img_name_to_pred_path[path.stem] = path

        benchmarks = {}
        for gt_path in filter(lambda p: p.stem in img_name_to_pred_path, gt_dir_path.glob("*.jpg")):
            img_name = gt_path.stem
            pred_path = img_name_to_pred_path[img_name]
            metric = self.benchmark(pred_path, gt_path, metric_names)
            benchmarks[img_name] = metric

        return benchmarks


default_metrics = Metrics()