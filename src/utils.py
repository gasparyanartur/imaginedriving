from pathlib import Path
from collections.abc import Iterable
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np


def get_device():
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def show_img(img: Tensor, save_path: Path = None):
    if isinstance(img, (list, tuple, dict, set)):
        img = list(img)
        for i in range(len(img)):
            img[i] = img[i].squeeze().to(img[0].device)

        img = torch.stack(img)

    img = img.detach().cpu()

    batch_size = len(img.shape)
    if batch_size == 4:
        fig, axes = plt.subplots(img.size(0), 1, figsize=(16, 4 * batch_size))

    elif batch_size == 3:
        fig = plt.figure()
        ax = plt.gca()

        axes = [ax]
        img = img[None, ...]

    else:
        assert False

    if not isinstance(axes, Iterable):
        axes = [axes]

    if img.shape[-1] != 3:
        img = img.permute(0, 2, 3, 1)

    for ax, img in zip(axes, img):
        ax.imshow(img)
        ax.axis("off")

    if save_path:
        fig.savefig(str(save_path))


def batch_if_not_iterable(item: any, single_dim: int = 3) -> Iterable[any]:
    if isinstance(item, (torch.Tensor, np.ndarray)):
        if len(item) == single_dim:
            item = item[None, ...]

        return item

    if not isinstance(item, Iterable):
        return [item]

    return item


def validate_same_len(*iters) -> None:
    prev_iter_len = None
    for iterator in iters:
        iter_len = len(iterator)

        if (prev_iter_len is not None) and (iter_len != prev_iter_len):
            raise ValueError(f"Expected same length on iterators, but received {[len(i) for i in iters]}")
        
        prev_iter_len = iter_len


def combine_kwargs(kwargs, extra_kwargs):
    extra_kwargs = extra_kwargs or {} 
    kwargs = dict(kwargs, **extra_kwargs)
    return kwargs