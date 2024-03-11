import pathlib as pl
from pathlib import Path
from collections.abc import Iterable
import pandas as pd

import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import read_image, load_img_if_path
from src.utils import batch_img_if_single


def benchmark_fid(fid_metric, pred: Tensor, gt: Tensor) -> Tensor:
    fid_metric.reset()
    fid_metric.update(pred, real=False)
    fid_metric.update(gt, real=True)
    fid = fid_metric.compute()
    return fid

fid_metric = FrechetInceptionDistance(feature=64, normalize=True)
psnr_metric = PeakSignalNoiseRatio(data_range=(0, 1))
ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0, 1))
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)

single_metrics = {
    "psnr": psnr_metric.__call__,
    "ssim": ssim_metric.__call__,
    "lpips": lpips_metric.__call__,
}

batch_metrics = {
    "fid": lambda pred, gt: benchmark_fid(fid_metric, pred, gt),
    "psnr": psnr_metric.__call__,
    "ssim": ssim_metric.__call__,
    "lpips": lpips_metric.__call__,
}

class Metrics:
    def __init__(self) -> None:
        ...

    @classmethod
    def get_default(cls) -> "Metrics":
        return default_metrics

    def benchmark_img_to_img(self, pred: Tensor | Path | str, gt: Tensor | Path | str, metric_names: Iterable[str] = single_metrics):
        # TODO: Change to pass image OR dataset

        def load_imgs(img):
            if isinstance(img, str):
                img = Path(img)

            if isinstance(img, Path):        
                if not img.exists():
                    raise ValueError(f"Could not find image in path")

                if img.is_dir():
                    imgs = []
                    for img_path in img.glob("*.jpg"):
                        try:
                            imgs.append(read_image(img_path))
                        except Exception:
                            continue

                    img = torch.stack(imgs) 

                elif img.is_file(): 
                    img = read_image(img)[None, ...]
            
            elif isinstance(img, Tensor):
                img = batch_img_if_single(img)

            return img

        pred = load_imgs(pred)
        gt = load_imgs(gt)

        metrics = {}

        for metric_name in metric_names:
            if not metric_name in single_metrics:
                raise ValueError(f"Metric {metric_name} not found in metrics {single_metrics.keys()}")

            with torch.no_grad():
                metric = single_metrics[metric_name](pred, gt)

            metrics[metric_name] = metric.item()

        return metrics

    def benchmark_dir_to_dir(
        self, pred_dir_path: Path, gt_dir_path: Path, metric_names: str | Iterable[str] = single_metrics
    ) -> dict[str, float]:
        # TODO: Combine with img_to_img and batch_to_batch

        img_name_to_pred_path = {}
        for path in pred_dir_path.glob("*.jpg"):
            img_name_to_pred_path[path.stem] = path

        benchmarks = {}
        for gt_path in filter(lambda p: p.stem in img_name_to_pred_path, gt_dir_path.glob("*.jpg")):
            img_name = gt_path.stem
            pred_path = img_name_to_pred_path[img_name]
            metric = self.benchmark_img_to_img(pred_path, gt_path, metric_names)
            benchmarks[img_name] = metric

        return benchmarks


def save_metrics(metrics, save_path: Path, print_results: bool = False):
    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.to_csv(save_path)

    if print_results:
        print(df)


default_metrics = Metrics()