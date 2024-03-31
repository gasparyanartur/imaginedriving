from typing import Any
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable
import logging

import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import DynamicDataset
from src.utils import batch_if_not_iterable, get_device


def get_mean_aggregate(
    single_function,
    preds: DynamicDataset,
    gts: DynamicDataset,
    mean_value: bool = True,
    match_attrs=("scene", "sample"),
    data_type="rgb",
    device=get_device(),
) -> float:
    running_value = torch.zeros(1, dtype=torch.float32, device=device)
    count = 0

    if isinstance(preds, DynamicDataset) or isinstance(gts, DynamicDataset):
        if not (isinstance(preds, DynamicDataset) and isinstance(gts, DynamicDataset)):
            raise ValueError(
                f"Expected both NamedImageDataset, but got {type(preds)} and {type(gts)}"
            )

        sample_iter = preds.get_matching(gts, match_attrs=match_attrs)
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


class BaseMetric(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name


class SingleMetric(BaseMetric, ABC):
    @abstractmethod
    def execute_single(self, pred: dict[str, Any], gt: dict[str, Any]) -> float:
        raise NotImplementedError

    def execute_map(
        self,
        preds: DynamicDataset,
        gts: DynamicDataset,
        match_attrs=("scene", "sample"),
    ) -> dict[tuple[str], float]:
        if not isinstance(preds, DynamicDataset) and not isinstance(
            gts, DynamicDataset
        ):
            raise NotImplementedError

        metrics = {}
        for pred, gt in preds.get_matching(gts, match_attrs=match_attrs):
            query = tuple(pred[attr] for attr in match_attrs)
            metrics[query] = self.execute_single(pred, gt).item()

        return metrics


class AggregateMetric(BaseMetric, ABC):
    @abstractmethod
    def execute_aggregate(
        self,
        preds: DynamicDataset,
        gts: DynamicDataset,
        match_attrs: tuple[str] = ("scene", "sample"),
    ) -> dict[tuple[str], float]:
        raise NotImplementedError


class DefaultMetricWrapper(SingleMetric, AggregateMetric):
    def __init__(
        self, name: str, model, device=get_device(), require_batch: bool = True, data_type: str = "rgb"
    ) -> None:
        super().__init__(name)
        self.model = model.to(device)
        self.require_batch: bool = require_batch
        self.data_type = data_type

    def execute_single(
        self, pred: dict[str, Any], gt: dict[str, Any], device=get_device()
    ) -> float:
        pred = pred[self.data_type].to(device)
        gt = gt[self.data_type].to(device)

        if self.require_batch:
            pred = batch_if_not_iterable(pred)
            gt = batch_if_not_iterable(gt)

        return self.model(pred, gt)

    def execute_aggregate(
        self,
        preds: DynamicDataset,
        gts: DynamicDataset,
        match_attrs: tuple[str] = ("scene", "sample"),
    ) -> dict[tuple[str], float]:
        return get_mean_aggregate(
            lambda pred, gt: self.execute_single(pred, gt),
            preds,
            gts,
            mean_value=True,
            data_type=self.data_type,
            match_attrs=match_attrs,
        )


class PSNRMetric(DefaultMetricWrapper):
    def __init__(
        self,
        name="psnr",
        data_range=(0, 1),
        device=get_device(),
        data_type: str = "rgb",
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            PeakSignalNoiseRatio(data_range=data_range),
            device=device,
            data_type=data_type,
            **kwargs,
        )


class SSIMMetric(DefaultMetricWrapper):
    def __init__(
        self,
        name="ssim",
        data_range=(0, 1),
        device=get_device(),
        data_type: str = "rgb",
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            model=StructuralSimilarityIndexMeasure(data_range=data_range),
            device=device,
            data_type=data_type,
            **kwargs,
        ),


class LPIPSMetric(DefaultMetricWrapper):
    def __init__(
        self,
        name="lpips",
        net_type: str = "squeeze",
        normalize=True,
        device=get_device(),
        data_type: str = "rgb",
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            LearnedPerceptualImagePatchSimilarity(
                net_type=net_type, normalize=normalize, **kwargs
            ),
            device=device,
            data_type=data_type
        )


class FIDMetric(AggregateMetric):
    def __init__(
        self,
        name: str = "fid",
        feature: int = 64,
        normalize: bool = True,
        device=get_device(),
        data_type: str = "rgb",
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.model = (
            FrechetInceptionDistance(feature=feature, normalize=normalize, **kwargs)
            .set_dtype(torch.float64)
            .to(device)
        )
        self.data_type = data_type

    def execute_aggregate(
        self,
        preds: DynamicDataset,
        gts: DynamicDataset,
        match_attrs=("scene",),
        device=get_device(),
    ) -> float:
        self.model.reset()

        matching_gts = preds.get_matching(gts, match_attrs=match_attrs)
        for _, gt in matching_gts:
            gt = gt[self.data_type].to(device)
            gt = batch_if_not_iterable(gt).double()
            self.model.update(gt, real=True)

        for pred in preds:
            pred = pred[self.data_type].to(device)
            pred = batch_if_not_iterable(pred).double()
            self.model.update(pred, real=False)

        return self.model.compute().item()


default_aggregate_metrics = (PSNRMetric(), SSIMMetric(), LPIPSMetric(), FIDMetric())

default_single_metrics = (
    PSNRMetric(),
    SSIMMetric(),
    LPIPSMetric(),
)


def run_benchmark_suite(
    preds: DynamicDataset,
    gts: DynamicDataset,
    aggregate_metrics: list[AggregateMetric] = default_aggregate_metrics,
    single_metrics: list[SingleMetric] = default_single_metrics,
):
    metrics = {}

    with torch.no_grad():
        metrics["aggregate"] = {
            aggregate_metric.name: aggregate_metric.execute_aggregate(preds, gts)
            for aggregate_metric in aggregate_metrics
        }

        # Go from query: metrics to query[1]/query[2]: ...: sample: metrics
        metrics["single"] = {}

        for single_metric in single_metrics:
            values = single_metric.execute_map(preds, gts)
            new_values = {"/".join(query): metric for query, metric in values.items()}

            metrics["single"][single_metric.name] = new_values

    return metrics
