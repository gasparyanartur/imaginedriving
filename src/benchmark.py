import pathlib as pl
from collections.abc import Iterable
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import read_image

metric_names_mapping = {
    "psnr": PeakSignalNoiseRatio(data_range=1.0),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
    "lpips": LearnedPerceptualImagePatchSimilarity(net_type="squeeze"),
    "fid": FrechetInceptionDistance(feature=64),
}


def benchmark_img_to_img(
    img_pred_path: pl.Path, img_gt_path: pl.Path, metric_names: Iterable[str]
) -> dict[str, float]:

    img_pred = read_image(img_pred_path)[None, ...]
    img_gt = read_image(img_gt_path)[None, ...]

    metrics = {}

    for metric_name in metric_names:
        if metric_name in metric_names_mapping:
            if metric_name == "fid":
                metric_names_mapping[metric_name].update(img_pred, real=False)
                metric_names_mapping[metric_name].update(img_gt, real=True)
                metrics[metric_name] = metric_names_mapping[metric_name].compute()

            else:
                metrics[metric_name] = metric_names_mapping[metric_name](
                    img_pred, img_gt)

    return metrics


def benchmark_dir_to_dir(
    pred_dir_path: pl.Path, gt_dir_path: pl.Path, metric_names: Iterable[str]
) -> dict[str, float]:
    pred_paths = {}
    benchmarks = {}
    for path in pred_dir_path.glob("*.jpg"):
        pred_paths[path.stem] = path

    for path in gt_dir_path.glob("*.jpg"):
        if path.stem in pred_paths:
            benchmarks[path.stem] = benchmark_img_to_img(
                path, pred_paths[path.stem], metric_names
            )

    return benchmarks
