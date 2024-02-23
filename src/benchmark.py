from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import pathlib as pl
from collections.abc import Iterable

from src.data import load_img_paths_from_dir

metric_names_mapping = {"psnr": PeakSignalNoiseRatio(data_range=1.0),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
                "lpips": LearnedPerceptualImagePatchSimilarity(et_type='squeeze')
                }

def benchmark_img_to_img(img_pred_path: pl.Path, img_gt_path: pl.Path, metric_names: Iterable) -> dict():
    
    img_pred = load_img_paths_from_dir(img_pred_path)
    img_gt = load_img_paths_from_dir(img_gt_path)


    metrics = {}

    for metric_name in metric_names:
        if metric_name in metric_names_mapping:
            metrics[metric_name] = metric_names[metric_name](img_pred, img_gt)
    
    return metrics
