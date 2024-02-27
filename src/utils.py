import torch
from torch import Tensor
import matplotlib.pyplot as plt


def get_device():
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def show_img(img: Tensor, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    if img.shape[-1] != 3:
        img = img.permute(1, 2, 0)

    ax.imshow(img)
    ax.axis("off")
