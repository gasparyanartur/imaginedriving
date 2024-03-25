from collections.abc import Iterable
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import v2 as tvtf2
import torchvision


base_img_pipeline = tvtf2.Compose([tvtf2.ToDtype(torch.float32, scale=True)])




def img_float_to_img(img: Tensor):
    img = img * 255
    img = img.to(device=img.device, dtype=torch.uint8)
    return img


def sort_paths_numerically(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: int(path.stem))


def load_json(path):
    with open(path, "rb") as f:
        return json.load(f)


def load_img_paths_from_dir(dir_path: Path):
    img_paths = list(sorted(dir_path.glob("*.jpg")))
    return img_paths


def read_image(img_path: Path, tf_pipeline: tvtf2.Compose = base_img_pipeline) -> Tensor:
    img = torchvision.io.read_image(str(img_path))
    img = tf_pipeline(img)

    return img


def save_image(save_path: Path, img: Tensor, jpg_quality: int = 100) -> None:
    img = img.detach().cpu()
    img = img.squeeze()

    if torch.is_floating_point(img):
        img = img_float_to_img(img)

    torchvision.io.write_jpeg(img, str(save_path), quality=jpg_quality)


def load_img_if_path(img: str | Path | Tensor) -> Tensor:
    if isinstance(img, str):
        img = Path(img)

    if isinstance(img, Path):
        img = read_image(img)

    return img


class MemoryImageDataset(Dataset):
    def __init__(self, imgs: dict[str, Tensor], device=None) -> None:
        super().__init__()
        self.imgs = imgs
        self.names = list(imgs.keys())
        self.device = device

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        img = self.imgs[key]
        if self.device:
            img = img.to(self.device)

        return img


class LazyImageDataset(Dataset):
    def __init__(self, paths: dict[str, Path], device=None) -> None:
        super().__init__()
        self.paths = paths
        self.names = list(paths.keys())
        self.device = device

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            key = self.names[key]

        path = self.paths[key]
        img = read_image(path)
        if self.device:
            img = img.to(self.device)

        return img


class NamedImageDataset(Dataset):
    def __init__(self, names: Iterable[str], imgs: Dataset | list[Tensor], device=None):
        if not isinstance(imgs, Dataset):
            if isinstance(imgs, Tensor) and not len(imgs.shape) == 4:
                raise ValueError(f"Need to have a batch, received {len(imgs.shape)}")

            if len(imgs) != len(names):
                raise ValueError(
                    f"names and imgs need to have the same length, received {len(names)}, {len(imgs)}"
                )

            imgs = MemoryImageDataset({name: img for name, img in zip(names, imgs)})

        self.names = list(names)
        self.imgs = imgs
        self.device = device

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


class DirectoryDataset(NamedImageDataset):
    def __init__(
        self,
        dir_path: Path,
        names: Iterable[str],
        imgs: Dataset | list[Tensor],
        name: str = None,
        device=None,
    ):
        super().__init__(names=names, imgs=imgs, device=device)
        self.dir_path = dir_path
        self.name = name

    @classmethod
    def from_directory(
        cls, dir_path: Path, in_memory: bool = False, name: str = None, device=None
    ) -> "NamedImageDataset":
        if name is None:
            name = dir_path.stem

        paths = load_img_paths_from_dir(dir_path)
        names = [path.stem for path in paths]

        if in_memory:
            imgs = MemoryImageDataset(
                imgs={name: read_image(path) for name, path in zip(names, paths)},
                device=device,
            )

        else:
            imgs = LazyImageDataset(
                paths={name: path for name, path in zip(names, paths)}, device=device
            )

        return DirectoryDataset(
            dir_path=dir_path, names=names, imgs=imgs, name=name, device=device
        )



class PandasetDataset(Dataset):
    def __init__(self, dataset_path: Path, scenes: Iterable[str], text_embedding: Tensor, scene_to_img_dir: str = "camera/front_camera", preprocessing="train") -> None:
        super().__init__()

        self.text_embedding = text_embedding
        img_paths = []

        for scene in scenes:
            img_dir_path = dataset_path / scene / scene_to_img_dir
            img_paths.extend(load_img_paths_from_dir(img_dir_path))

        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx])
        return {
            "pixel_values": img,
            "text-embedding": self.text_embedding
        }
