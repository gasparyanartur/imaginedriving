from pathlib import Path
from collections.abc import Iterable
import matplotlib.pyplot as plt
import torch
from torch import Tensor


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
