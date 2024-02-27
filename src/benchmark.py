from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import pathlib as pl
from collections.abc import Iterable

from src.data import read_image

metric_names_mapping = {
    "psnr": PeakSignalNoiseRatio(data_range=1.0),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
    "lpips": LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
}

def benchmark_img_to_img(img_pred_path: pl.Path, img_gt_path: pl.Path, metric_names: Iterable[str]) -> dict[str, float]:
    
    img_pred = read_image(img_pred_path)
    img_gt = read_image(img_gt_path)


    metrics = {}

    for metric_name in metric_names:
        if metric_name in metric_names_mapping:
            metrics[metric_name] = metric_names_mapping[metric_name](img_pred, img_gt)
    
    return metrics


def benchmark_dir_to_dir(dir1_path: pl.Path, dir2_path: pl.Path, metric_names: Iterable[str]) -> dict[str, float]:
    dir1 = {}
    dir2 = {}
    for f in dir1_path.glob("*.jpg"):
        dir1[f.stem] = f
    print(dir1)

    for f in dir2_path.glob("*.jpg"):
        if f.stem in dir1:
            dir2[f.stem] = benchmark_img_to_img(f, dir1[f.stem], metric_names)

    return dir2
