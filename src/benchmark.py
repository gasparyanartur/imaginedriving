from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import logging

import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import NamedImageDataset
from src.utils import batch_img_if_single


def benchmark_fids(fid_metric, pred: Tensor, gt: Tensor) -> Tensor:
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
    "fid": lambda preds, gts: benchmark_fids(fid_metric, preds, gts),
    "psnr": psnr_metric.__call__,
    "ssim": ssim_metric.__call__,
    "lpips": lpips_metric.__call__,
}


def get_mean_aggregate(
    single_function,
    preds: NamedImageDataset | Iterable[Tensor] | Tensor,
    gts: NamedImageDataset | Iterable[Tensor] | Tensor,
    mean_value: bool = True,
) -> torch.Tensor:
    running_value = torch.zeros(1, dtype=torch.float32)
    count = 0

    if isinstance(preds, NamedImageDataset) or isinstance(gts, NamedImageDataset):
        if not (
            isinstance(preds, NamedImageDataset) and isinstance(gts, NamedImageDataset)
        ):
            raise ValueError(f"Expected both NamedImageDataset, but got {preds} {gts}")

        sample_iter = ((pred, gt) for _, (pred, gt) in preds.get_matching(gts))
    else:
        sample_iter = zip(preds, gts)

    for pred, gt in sample_iter:
        running_value += single_function(pred, gt)
        count += 1

    if count == 0:
        logging.warning("Could not find any matching values")
        return torch.nan

    if mean_value:
        running_value /= count

    return running_value.item()


class Metric(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name


class SingleMetric(Metric, ABC):
    @abstractmethod
    def execute_single(self, pred: Tensor, gt: Tensor) -> float:
        raise NotImplementedError

    def execute_map(
        self,
        preds: NamedImageDataset | Iterable[Tensor] | Tensor,
        gts: NamedImageDataset | Iterable[Tensor] | Tensor,
    ) -> pd.DataFrame:
        if not isinstance(preds, NamedImageDataset) and not isinstance(
            gts, NamedImageDataset
        ):
            raise NotImplementedError

        metrics = {
            sample_name: self.execute_single(pred, gt).item()
            for sample_name, (pred, gt) in preds.get_matching(gts)
        }
        return pd.Series(metrics)


class AggregateMetric(Metric, ABC):
    @abstractmethod
    def execute_aggregate(
        self, preds: NamedImageDataset, gts: NamedImageDataset
    ) -> pd.DataFrame:
        raise NotImplementedError


class DefaultMetricWrapper(SingleMetric, AggregateMetric):
    def __init__(self, name, model, require_batch: bool = True) -> None:
        super().__init__(name)
        self.model = model
        self.require_batch: bool = require_batch

    def execute_single(self, pred: Tensor, gt: Tensor) -> float:
        if self.require_batch:
            pred = batch_img_if_single(pred)
            gt = batch_img_if_single(gt)

        return self.model(pred, gt)

    def execute_aggregate(
        self,
        preds: NamedImageDataset | Iterable[Tensor] | Tensor,
        gts: NamedImageDataset | Iterable[Tensor] | Tensor,
    ) -> pd.DataFrame:
        def func(pred, gt):
            return self.execute_single(pred, gt)

        return get_mean_aggregate(func, preds, gts, mean_value=True)


class PSNRMetric(DefaultMetricWrapper):
    def __init__(self, name="psnr", data_range=(0, 1), **kwargs) -> None:
        super().__init__(name, PeakSignalNoiseRatio(data_range=data_range, **kwargs))


class SSIMMetric(DefaultMetricWrapper):
    def __init__(self, name="ssim", data_range=(0, 1), **kwargs) -> None:
        super().__init__(
            name,
            model=StructuralSimilarityIndexMeasure(data_range=data_range, **kwargs),
        )


class LPIPSMetric(DefaultMetricWrapper):
    def __init__(
        self, name="lpips", net_type: str = "squeeze", normalize=True, **kwargs
    ) -> None:
        super().__init__(
            name,
            LearnedPerceptualImagePatchSimilarity(
                net_type=net_type, normalize=normalize, **kwargs
            ),
        )


class FIDMetric(AggregateMetric):
    def __init__(self, name="fid", feature=64, normalize=True, **kwargs) -> None:
        super().__init__(name)
        self.model = FrechetInceptionDistance(
            feature=feature, normalize=normalize, **kwargs
        ).set_dtype(torch.float64)

    def execute_aggregate(
        self, preds: NamedImageDataset, gts: NamedImageDataset
    ) -> pd.DataFrame:
        self.model.reset()

        for _, gt in gts:
            gt = batch_img_if_single(gt).double()
            self.model.update(gt, real=True)

        for _, pred in preds:
            pred = batch_img_if_single(pred).double()
            self.model.update(pred, real=False)

        return self.model.compute().item()


default_aggregate_metrics = (PSNRMetric(), SSIMMetric(), LPIPSMetric(), FIDMetric())

default_single_metrics = (
    PSNRMetric(),
    SSIMMetric(),
    LPIPSMetric(),
)


def benchmark_aggregate_metrics(
    preds: NamedImageDataset,
    gts: NamedImageDataset,
    metrics: list[AggregateMetric] = default_aggregate_metrics,
) -> pd.DataFrame:
    with torch.no_grad():
        results = {
            metric.name: [metric.execute_aggregate(preds, gts)] for metric in metrics
        }
        return pd.DataFrame.from_dict(results, orient="columns")


def benchmark_single_metrics(
    preds: NamedImageDataset,
    gts: NamedImageDataset,
    metrics: list[SingleMetric] = default_single_metrics,
) -> pd.DataFrame:
    if len(metrics) == 0:
        return []

    with torch.no_grad():
        if len(metrics) == 1:
            return metrics[0].execute_map(preds, gts)

        df = pd.concat([metric.execute_map(preds, gts) for metric in metrics], axis=1)
        df.columns = [metric.name for metric in metrics]
        return df
