from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
import json

import torch
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


class ImageContainer(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> Tensor:
        raise NotImplementedError


class MemoryImageContainer(ImageContainer):
    def __init__(self, imgs: dict[str, Tensor]) -> None:
        super().__init__()
        self.imgs = imgs
        self.names = list(imgs.keys())

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        img = self.imgs[key]
        return img


class LazyImageContainer(ImageContainer):
    def __init__(self, paths: dict[str, Path]) -> None:
        super().__init__()
        self.paths = paths
        self.names = list(paths.keys())

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        path = self.paths[key]
        img = read_image(path)

        return img


class ImageDataset:
    def __init__(self, names: Iterable[str], imgs: ImageContainer | list[Tensor]):
        if not isinstance(imgs, ImageContainer):
            imgs = MemoryImageContainer({name: img for name, img in zip(names, imgs)})
        else:
            print("NOT LIST")

        self.names = list(names)
        self.imgs = imgs

    @classmethod
    def from_directory(cls, dir_path: Path, in_memory: bool = False) -> "ImageDataset":
        paths = load_img_paths_from_dir(dir_path)
        names = [path.stem for path in paths]

        if in_memory:
            imgs = MemoryImageContainer({name: read_image(path) for name, path in zip(names, paths)})

        else:
            imgs = LazyImageContainer({name: path for name, path in zip(names, paths)})

        return ImageDataset(names, imgs)
        
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, key: str | int) -> Tensor:
        if isinstance(key, int):
            name = self.names[key]

        else:
            name = key

        img = self.imgs[name]
        return {
            "name": name,
            "image": img
        }
