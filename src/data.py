from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import v2 as tvtf2
import torchvision


base_img_pipeline = tvtf2.Compose([tvtf2.ToDtype(torch.float32, scale=True)])


def sort_paths_numerically(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: int(path.stem))


def load_json(path):
    with open(path, "rb") as f:
        return json.load(f)


def load_img_paths_from_dir(dir_path: Path):
    img_paths = list(sorted(dir_path.glob("*.jpg")))
    return img_paths


def read_image(img_path: Path, pipeline_type: str = "base") -> Tensor:
    img = torchvision.io.read_image(str(img_path))

    if pipeline_type == "base":
        img = base_img_pipeline(img)

    else:
        raise NotImplementedError

    return img


def save_image(save_path: Path, img: Tensor, jpg_quality: int = 100) -> None:
    torchvision.io.write_jpeg(img, str(save_path), quality=jpg_quality)


def load_img_if_path(img: str | Path | Tensor) -> Tensor:
    if isinstance(img, str):
        img = Path(img)

    if isinstance(img, Path):
        img = read_image(img)

    return img


class MemoryImageDataset(Dataset):
    def __init__(self, imgs: dict[str, Tensor]) -> None:
        super().__init__()
        self.imgs = imgs
        self.names = list(imgs.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        img = self.imgs[key]
        return img


class LazyImageDataset(Dataset):
    def __init__(self, paths: dict[str, Path]) -> None:
        super().__init__()
        self.paths = paths
        self.names = list(paths.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        path = self.paths[key]
        img = read_image(path)

        return img


class NamedImageDataset:
    def __init__(self, names: Iterable[str], imgs: Dataset | list[Tensor]):
        if not isinstance(imgs, Dataset):
            if isinstance(imgs, Tensor) and not len(imgs.shape) == 4:
                raise ValueError(f"Need to have a batch, received {len(imgs.shape)}")

            if len(imgs) != len(names):
                raise ValueError(
                    f"names and imgs need to have the same length, received {len(names)}, {len(imgs)}"
                )

            imgs = MemoryImageDataset({name: img for name, img in zip(names, imgs)})

        else:
            print("NOT LIST")

        self.names = list(names)
        self.imgs = imgs

    @classmethod
    def from_directory(
        cls, dir_path: Path, in_memory: bool = False
    ) -> "NamedImageDataset":
        paths = load_img_paths_from_dir(dir_path)
        names = [path.stem for path in paths]

        if in_memory:
            imgs = MemoryImageDataset(
                {name: read_image(path) for name, path in zip(names, paths)}
            )

        else:
            imgs = LazyImageDataset({name: path for name, path in zip(names, paths)})

        return NamedImageDataset(names, imgs)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, key: str | int) -> tuple[str, Tensor] | Tensor:
        if isinstance(key, int):
            name = self.names[key]
            img = self.imgs[name]
            return name, img

        else:
            return self.imgs[key]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_matching_names(self, other: "NamedImageDataset") -> list[str]:
        return list(set(self.names).intersection(other.names))

    def get_matching(self, other: "NamedImageDataset"):
        for name in self.get_matching_names(other):
            yield name, (self[name], other[name])
