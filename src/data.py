from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterable, Generator
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import v2 as tvtf2
import torchvision

from src.configuration import read_yaml


base_img_pipeline = tvtf2.Compose([tvtf2.ToDtype(torch.float32, scale=True)])
pandaset_img_suffix = ".jpg"

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


def iter_numeric_names(start_name: str | int, end_name: str | int, fixed_len: int = 2):
    start_name = int(start_name)
    end_name = int(end_name)

    for i in range(start_name, end_name+1):
        num = str(i)
        if fixed_len:
            num = num.rjust(fixed_len, "0")
        
        yield num


def expand_data_tree(data_tree, scene_name_len: int = 3, sample_name_len: int = 2):
    dataset_dict = {}
    for dataset_name, dataset in data_tree.items():
        scene_dict = {}
        for scene_name, scene in dataset.items():
            if isinstance(scene, str):
                assert scene == "*"
                sample_list = None
                
            else:
                sample_list = []
                for sample in scene:
                    sample = str(sample)
                    sample = sample.replace(" ", "")
                    if sample.isdigit():
                        sample = sample.rjust(sample_name_len, "0")
                        sample_list.append(sample)
                    else:
                        assert "-" in sample
                        sample_from, sample_to = sample.split("-")
                        assert sample_from.isdigit() and sample_to.isdigit()
                        sample_list.extend(iter_numeric_names(sample_from, sample_to, fixed_len=sample_name_len))

            scene_name = str(scene_name)
            if scene_name.isdigit():
                scene_name = scene_name.rjust(scene_name_len, "0")
                scene_dict[scene_name] = sample_list
            else:
                assert "-" in scene_name
                scene_from, scene_to = scene_name.split("-")
                assert scene_from.isdigit() and scene_to.isdigit()

                for new_scene_name in iter_numeric_names(scene_from, scene_to, fixed_len=scene_name_len):
                    scene_dict[new_scene_name] = sample_list

        dataset_dict[dataset_name] = scene_dict
    return dataset_dict


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
    def __init__(self, dataset_path: Path, scenes: Iterable[str], text_embedding: Tensor, scene_to_img_dir: str = "camera/front_camera", preprocessing=base_img_pipeline) -> None:
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


class DataGetter(ABC):
    @abstractmethod
    def get_sample_path(self, dataset_path: Path, scene: str, sample: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def iterate_samples_in_scene(self, dataset_path: Path, scene: str) -> Generator[Path]:
        raise NotImplementedError

    @abstractmethod
    def load_data(self, path: Path):
        raise NotImplementedError

    def paths_from_tree(self, dataset_path: Path, expanded_tree: dict[str, any]) -> list[Path]:
        samples = []
        for _, dataset_dict in expanded_tree.items():
            for scene_name, sample_list in dataset_dict.items():
                if sample_list is None:
                    samples.extend(self.iterate_samples_in_scene(dataset_path, scene_name))
                    continue

                for sample_name in sample_list:
                    samples.append(self.get_sample_path(dataset_path, scene_name, sample_name))
        return samples


class PandasetImageGetter(DataGetter):
    def __init__(self, camera: str = "front_camera", tf_pipeline=base_img_pipeline):
        self.camera = camera
        self.tf_pipeline = tf_pipeline

    def get_sample_path(self, dataset_path: Path, scene: str, sample: str) -> Path:
        path = dataset_path / scene / "camera" / self.camera / sample
        path = path.with_suffix(pandaset_img_suffix)
        return path

    def iterate_samples_in_scene(self, dataset_path: Path, scene: str) -> Generator[Path]:
        cam_path = dataset_path / scene / "camera" / self.camera
        return cam_path.glob(f"*{pandaset_img_suffix}")

    def load_data(self, path: Path):
        img = read_image(path, self.tf_pipeline)
        return img


class SampleLoader(ABC):
    @abstractmethod
    def load_sample(self, path, **kwargs):
        raise NotImplementedError


class DynamicDataset(Dataset):      # Dataset / Scene / Sample
    def __init__(self, dataset_path: Path, tree_path: Path, data_getters: dict[str, DataGetter]):
        data_tree = read_yaml(tree_path)
        expanded_tree = expand_data_tree(data_tree)
        self.sample_paths = {
            name: getter.paths_from_tree(dataset_path, expanded_tree) for name, getter in data_getters.items()
        }
        self.sample_names = list(self.sample_paths.keys())
        self.data_getters = data_getters

    def __len__(self) -> int:
        example_paths = next(self.sample_paths.values())
        return len(example_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = {} 

        for name in self.sample_names:
            getter = self.data_getters[name]
            path = self.sample_paths[name][idx]
            data = getter.load_data(path)

            sample[name] = data

        return sample

