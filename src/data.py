from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections.abc import Iterable, Generator, Callable
import os
from pathlib import Path
import json
import yaml

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import v2 as tvtf2
import torchvision
import logging

from src.utils import get_env, set_env, set_if_no_key


base_img_pipeline = tvtf2.Compose([tvtf2.ToDtype(torch.float32, scale=True)])

suffixes = {("rgb", "pandaset"): ".jpg", ("rgb", "neurad"): ".jpg"}


def get_dataset_from_path(path: Path) -> str:
    return path.stem


def save_yaml(path: Path, data: dict[str, Any]):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def setup_project(config_path: Path = None):
    logging.getLogger().setLevel(logging.INFO)

    if not torch.cuda.is_available():
        logging.warning(
            f"CUDA not detected. Running on CPU. The code is not supported for CPU and will most likely give incorrect results. Proceed with caution."
        )

    proj_dir = get_env("PROJ_DIR") or Path.cwd()

    if config_path is None:
        config_path = get_env("PROJ_CONF_PATH")

        if config_path is None:
            config_path = proj_dir / "proj_config.yml"

            if not config_path.exists():
                raise ValueError(f"No config path specified")

    config = read_yaml(config_path)

    proj_dir = set_if_no_key(config, "PROJ_DIR", proj_dir)
    cache_dir = set_if_no_key(config, "CACHE_DIR", proj_dir / ".cache")
    cache_dir = config["CACHE_DIR"]

    set_env("HF_HUB_CACHE", cache_dir / "hf")  # Huggingface cache dir
    set_env("MPLCONFIGDIR", cache_dir / "mpl")  # Matplotlib cache dir

    return config


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


def read_image(
    img_path: Path, tf_pipeline: tvtf2.Compose = base_img_pipeline
) -> Tensor:
    img = torchvision.io.read_image(str(img_path))
    img = tf_pipeline(img)

    return img


def save_image(save_path: Path, img: Tensor, jpg_quality: int = 100) -> None:
    img = img.detach().cpu()
    img = img.squeeze()

    if torch.is_floating_point(img):
        img = img_float_to_img(img)

    torchvision.io.write_jpeg(img, str(save_path), quality=jpg_quality)


def save_json(save_path: Path, data: dict[str, Any]) -> None:
    with open(save_path, "w") as f:
        json.dump(data, f)


def load_img_if_path(img: str | Path | Tensor) -> Tensor:
    if isinstance(img, str):
        img = Path(img)

    if isinstance(img, Path):
        img = read_image(img)

    return img


def iter_numeric_names(start_name: str | int, end_name: str | int, fixed_len: int = 2):
    start_name = int(start_name)
    end_name = int(end_name)

    for i in range(start_name, end_name + 1):
        num = str(i)
        if fixed_len:
            num = num.rjust(fixed_len, "0")

        yield num


def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_data_tree(
    data_tree: dict | Path, scene_name_len: int = 3, sample_name_len: int = 2
):
    if isinstance(data_tree, (str, Path)):
        data_tree = Path(data_tree)
        data_tree = read_yaml(data_tree)

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
                        sample_list.extend(
                            iter_numeric_names(
                                sample_from, sample_to, fixed_len=sample_name_len
                            )
                        )

            scene_name = str(scene_name)
            if scene_name.isdigit():
                scene_name = scene_name.rjust(scene_name_len, "0")
                scene_dict[scene_name] = sample_list
            else:
                assert "-" in scene_name
                scene_from, scene_to = scene_name.split("-")
                assert scene_from.isdigit() and scene_to.isdigit()

                for new_scene_name in iter_numeric_names(
                    scene_from, scene_to, fixed_len=scene_name_len
                ):
                    scene_dict[new_scene_name] = sample_list

        dataset_dict[dataset_name] = scene_dict
    return dataset_dict


@dataclass
class SampleInfo:
    dataset: str
    scene: str
    sample: str


class InfoGetter(ABC):
    def __init__(self, dataset_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    @abstractmethod
    def get_sample_names_in_scene(
        self, dataset_path: Path, scene: str, specs: dict[str, Any] = None
    ) -> Generator[str]:
        raise NotImplementedError

    @abstractmethod
    def get_path(self, dataset_path: Path, info: SampleInfo, specs: dict[str, Any]):
        raise NotImplemented

    def parse_tree(
        self, dataset_path: Path, data_tree: dict[str, any]
    ) -> list[SampleInfo]:
        samples = []

        for dataset_name, dataset_dict in data_tree.items():
            for scene_name, sample_list in dataset_dict.items():
                if sample_list is None:
                    sample_list = self.get_sample_names_in_scene(
                        dataset_path, scene_name
                    )

                for sample_name in sample_list:
                    info = SampleInfo(dataset_name, scene_name, sample_name)
                    samples.append(info)

        return samples


class PandasetInfoGetter(InfoGetter):
    def __init__(self) -> None:
        super().__init__("pandaset")

    def _get_sample_dir_path(self, dataset_path, scene, specs=None):
        if specs is None:
            data_type = "rgb"
            camera = "front_camera"

        else:
            data_type = specs.get("data_type", "rgb")
            camera = specs.get("camera", "front_camera")

        match data_type:
            case "rgb":
                return dataset_path / scene / "camera" / camera

            case "lidar":
                return dataset_path / scene / "lidar"

            case _:
                raise NotImplementedError

    def get_sample_names_in_scene(
        self, dataset_path: Path, scene: str, specs: dict[str, Any] = None
    ) -> Generator[str]:
        sample_dir_path = self._get_sample_dir_path(dataset_path, scene, specs)

        data_type = "rgb" if specs is None else specs.get("data_type", "rgb")
        suffix = suffixes[data_type, self.dataset_name]

        for sample_path in sample_dir_path.glob(f"*{suffix}"):
            yield sample_path.stem

    def get_path(
        self, dataset_path: Path, info: SampleInfo, specs: dict[str, Any]
    ) -> Path:
        assert isinstance(info.sample, str)

        sample_dir_path = self._get_sample_dir_path(dataset_path, info.scene, specs)
        data_type = "rgb" if specs is None else specs.get("data_type", "rgb")
        suffix = suffixes[data_type, self.dataset_name]

        sample_path = (sample_dir_path / info.sample).with_suffix(suffix)
        return sample_path


class NeuRADInfoGetter(InfoGetter):
    def __init__(self):
        super().__init__("neurad")

    def _get_sample_dir_path(self, dataset_path: Path, scene: str, specs: dict[str, Any] = None) -> Path:
        if specs is None:
            shift = "0meter"
            data_type = "rgb"
            split = "test"
            camera = "front_camera"

        else:
            shift = specs.get("shift", "0meter")
            data_type = specs.get("data_type", "rgb")
            split = specs.get("split", "test")
            camera = specs.get("camera", "front_camera")

        match data_type:
            case "rgb":
                return dataset_path / scene / shift / split / data_type / camera
                
            case _:
                raise NotImplementedError


    def get_sample_names_in_scene(
        self, dataset_path: Path, scene: str, specs: dict[str, Any] = None
    ) -> Generator[str]:
        sample_dir_path = self._get_sample_dir_path(dataset_path, scene, specs)

        data_type = "rgb" if specs is None else specs.get("data_type", "rgb")
        suffix = suffixes[data_type, self.dataset_name]

        for sample_path in sample_dir_path.glob(f"*{suffix}"):
            yield sample_path.stem

    def get_path(
        self, dataset_path: Path, info: SampleInfo, specs: dict[str, Any]
    ) -> Path:
        assert isinstance(info.sample, str)

        sample_dir_path = self._get_sample_dir_path(dataset_path, info.scene, specs)
        data_type = "rgb" if specs is None else specs.get("data_type", "rgb")
        suffix = suffixes[data_type, self.dataset_name]

        sample_path = (sample_dir_path / info.sample).with_suffix(suffix)
        return sample_path


class DataGetter(ABC):
    def __init__(self, info_getter: InfoGetter, data_spec: dict[str, Any]) -> None:
        super().__init__()
        self.info_getter = info_getter
        self.data_spec = data_spec

    def get_data_path(self, dataset_path: Path, info: SampleInfo) -> Path:
        return self.info_getter.get_path(dataset_path, info, self.data_spec)

    @abstractmethod
    def get_data(self, dataset_path: Path, info: SampleInfo):
        raise NotImplementedError


class RGBDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
        tf_pipeline=base_img_pipeline,
    ):
        if "data_type" not in data_spec:
            data_spec["data_type"] = "rgb"

        self.info_getter = info_getter
        self.data_spec = data_spec
        self.tf_pipeline = tf_pipeline

    def get_data(self, dataset_path: Path, info: SampleInfo) -> Tensor:
        return read_image(self.get_data_path(dataset_path, info), self.tf_pipeline)


info_getter_builders: dict[str, Callable[[], InfoGetter]] = {
    "pandaset": PandasetInfoGetter,
    "neurad": NeuRADInfoGetter,
}

data_getter_builders: dict[str, Callable[[InfoGetter, dict[str, Any]], DataGetter]] = {
    "rgb": RGBDataGetter
}


class DynamicDataset(Dataset):  # Dataset / Scene / Sample
    def __init__(
        self,
        dataset_path: Path,
        data_tree: dict[str, Any],
        info_getter: InfoGetter,
        data_getters: dict[str, DataGetter],
    ):
        self.sample_infos: list[SampleInfo] = info_getter.parse_tree(
            dataset_path, data_tree
        )
        self.info_getter = info_getter
        self.data_getters = data_getters
        self.dataset_path = dataset_path

    @classmethod
    def from_config(cls, config: Path | dict[str, Any]):
        if isinstance(config, Path):
            config = read_yaml(config)

        dataset_path = Path(config["path"])
        data_tree = read_data_tree(config["data_tree"])
        dataset_name = config["dataset"]

        info_getter_factory = info_getter_builders[dataset_name]
        info_getter = info_getter_factory()

        data_getters = {}

        for data_type, spec in config["data_getters"].items():
            data_getter_factory = data_getter_builders[data_type]
            data_getter = data_getter_factory(info_getter, spec)

            data_getters[data_type] = data_getter

        return DynamicDataset(dataset_path, data_tree, info_getter, data_getters)

    @property
    def name(self) -> str:
        return self.info_getter.dataset_name

    def iter_range(
        self, id: int = 0, id_start: int = 0, id_stop: int = 0, verbose: bool = True
    ):
        assert id_start <= id <= id_stop

        skip = id_stop - id_start + 1
        start = id - id_start
        stop = len(self)

        for i in range(start, stop, skip):
            if verbose:
                logging.info(f"Sample {i}/{stop} (skip: {skip})")

            yield self[i]

    def __len__(self) -> int:
        return len(self.sample_infos)

    def __iter__(self) -> Generator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = self.sample_infos[idx]
        sample = asdict(info)

        for data_type, getter in self.data_getters.items():
            data = getter.get_data(self.dataset_path, info)
            sample[data_type] = data

        return sample

    def iter_attrs(self, attrs: Iterable[str]) -> Generator[tuple[any]]:
        for sample in self:
            yield tuple(sample[attr] for attr in attrs)

    def get_matching(
        self, other: "DynamicDataset", match_attrs: Iterable[str] = ("scene", "sample")
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        match_dict = {}
        for sample_self in self:
            sample_query = tuple(sample_self[attr] for attr in match_attrs)
            match_dict[sample_query] = sample_self

        matches = []
        for sample_other in other:
            sample_query = tuple(sample_other[attr] for attr in match_attrs)
            if sample_query in match_dict:
                sample_self = match_dict[sample_query]
                matches.append((sample_self, sample_other))

        return matches
