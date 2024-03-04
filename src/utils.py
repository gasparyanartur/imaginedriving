import torch
from torch import Tensor
import matplotlib.pyplot as plt


def get_device():
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def show_img(img: Tensor, ax=None):
    batch_size = len(img.shape)
    if batch_size == 4:
        fig, axes = plt.subplots(img.size(0), 1, figsize=(16, 4 * batch_size))

    elif batch_size == 3:
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        axes = [ax]
        img = img[None, ...]

    else:
        assert False

    if img.shape[-1] != 3:
        img = img.permute(0, 2, 3, 1)

    for ax, img in zip(axes, img):
        ax.imshow(img)
        ax.axis("off")
