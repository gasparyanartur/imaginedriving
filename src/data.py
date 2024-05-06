from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections.abc import Iterable, Generator, Callable
import os
from pathlib import Path
import json
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchvision
torchvision.disable_beta_transforms_warning(); from torchvision.transforms import v2 as transform
import torchvision
import logging

from src.utils import get_env, set_env, set_if_no_key




norm_img_pipeline = transform.Compose([transform.ConvertImageDtype(torch.float32)])
suffixes = {
    ("rgb", "pandaset"): ".jpg",
    ("rgb", "neurad"): ".jpg",
    ("lidar", "pandaset"): ".pkl.gz", 
    ("lidar", "neurad"): ".pkl.gz",
    ("pose", "pandaset"): ".json",
    ("intrinsics", "pandaset"): ".json"
}
RANGE_SEP = ":"


def get_dataset_from_path(path: Path) -> str:
    return path.stem


def save_yaml(path: Path, data: dict[str, Any]):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def setup_project(config_path: Path):
    logging.getLogger().setLevel(logging.INFO)

    if not torch.cuda.is_available():
        logging.warning(
            f"CUDA not detected. Running on CPU. The code is not supported for CPU and will most likely give incorrect results. Proceed with caution."
        )

    if config_path is None:
        project_dir = get_env("PROJECT_DIR") or Path.cwd()
        config_path = project_dir / "proj_config.yml"
        if not config_path.exists():
            raise ValueError(f"No config path specified")
            
    else:
        if not config_path.exists():
            raise ValueError(f"Could not find config at specified path: {config_path}")


    config = read_yaml(config_path)
    project_dir = config.get("project_path", get_env("PROJECT_DIR") or Path.cwd())
    cache_dir = config.get("cache_dir", get_env("CACHE_DIR", project_dir / ".cache"))

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
    img_path: Path, tf_pipeline: transform.Compose = norm_img_pipeline
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
                if scene == "*":
                    sample_list = (None, None, None)
                    
                elif RANGE_SEP in scene:
                    sample_list = scene.strip().split(RANGE_SEP)
                    assert len(sample_list) == 3

                    start_range = int(sample_list[0]) if sample_list[0] != '' else None
                    end_range = int(sample_list[1]) if sample_list[1] != '' else None
                    skip_range = int(sample_list[2]) if sample_list[2] != '' else None

                    sample_list = (start_range, end_range, skip_range)

                else:
                    raise NotImplementedError

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

        existing_scenes = set(path.stem for path in dataset_path.iterdir())

        for dataset_name, dataset_dict in data_tree.items():
            for scene_name, sample_list in dataset_dict.items():
                if scene_name not in existing_scenes:
                    continue

                if isinstance(sample_list, tuple) and len(sample_list) == 3:
                    start_range, end_range, skip_range = sample_list
                    sample_list = sorted(self.get_sample_names_in_scene(
                        dataset_path, scene_name
                    ))

                    if start_range is None:
                        start_range = 0

                    if end_range is None:
                        end_range = len(sample_list)

                    if skip_range is None:
                        skip_range = 1

                    sample_list = [sample_list[i] for i in range(start_range, end_range, skip_range)]

                if not isinstance(sample_list, list):
                    raise NotImplementedError

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
            case "rgb" | "pose" | "intrinsics":
                return dataset_path / scene / "camera" / camera

            case "lidar":
                return dataset_path / scene / "lidar"

            case _:
                return None

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

        if data_type in {"rgb", "lidar"}:
            sample_path = (sample_dir_path / info.sample)

        elif data_type == "pose":
            sample_path = (sample_dir_path / "poses")

        elif data_type == "intrinsics":
            sample_path = (sample_dir_path / "intrinsics")

        suffix = suffixes[data_type, self.dataset_name]
        sample_path = sample_path.with_suffix(suffix)

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
    def __init__(self, info_getter: InfoGetter, data_spec: dict[str, Any], data_type: str) -> None:
        super().__init__()
        self.info_getter = info_getter
        self.data_spec = data_spec
        self.data_spec["data_type"] = data_type

    def get_data_path(self, dataset_path: Path, info: SampleInfo) -> Path:
        return self.info_getter.get_path(dataset_path, info, self.data_spec)

    @abstractmethod
    def get_data(self, dataset_path: Path, info: SampleInfo):
        raise NotImplementedError


class MetaDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "meta")

    def get_data(self, dataset_path: Path, info: SampleInfo):
        result = {
            **asdict(info)
        }
        return result


class PromptDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "prompt")
        
        match self.data_spec.get("type", "static"):
            case "static":
                self.positive_prompt = self.data_spec.get("positive_prompt", "")
                self.negative_prompt = self.data_spec.get("negative_prompt", "")

            case _:
                raise NotImplementedError

    def get_data(self, dataset_path: Path, info: SampleInfo):
        return {"positive_prompt": self.positive_prompt, "negative_prompt": self.negative_prompt}

class RGBDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "rgb")

        dtype = self.data_spec.get("dtype", torch.float32)
        rescale = self.data_spec.get("rescale", True)
        width = self.data_spec.get("width", 1920)
        height = self.data_spec.get("height", 1080)

        if width is None:
            width = height

        elif height is None:
            height = width

        self.base_transform = transform.Compose(
            [
                transform.ConvertImageDtype(dtype) if rescale else transform.ToDtype(dtype),
                transform.Resize((height, width))
            ]
        )
        self.extra_transform: transform.Compose = None

    def set_extra_transforms(self, *transforms: transform.Transform) -> None:
        self.extra_transform = transform.Compose(transforms)

    def get_data(self, dataset_path: Path, info: SampleInfo) -> Tensor:
        rgb = read_image(self.get_data_path(dataset_path, info), self.base_transform)

        if self.extra_transform:
            rgb = self.extra_transform(rgb)

        return rgb


class LidarDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "lidar")

        # TODO
        dtype = self.data_spec.get("dtype", torch.float32)
        rescale = self.data_spec.get("rescale", True)
        width = self.data_spec.get("width", 1920)
        height = self.data_spec.get("height", 1080)

        if width is None:
            width = height

        elif height is None:
            height = width

        
    def get_data(self, dataset_path: Path, info: SampleInfo) -> Tensor:
        path = self.get_data_path(dataset_path, info)
        data = pd.read_pickle(path)
        
        return data


class PoseDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "pose")

        
    def get_data(self, dataset_path: Path, info: SampleInfo) -> dict[str, dict[str, float]]:
        file_path = self.get_data_path(dataset_path, info)
        poses = load_json(file_path)
        pose_idx = int(info.sample)
        pose = poses[pose_idx]
        return pose


class IntrinsicsDataGetter(DataGetter):
    def __init__(
        self,
        info_getter: InfoGetter,
        data_spec: dict[str, Any],
    ):
        super().__init__(info_getter, data_spec, "intrinsics")

        
    def get_data(self, dataset_path: Path, info: SampleInfo) -> dict[str, dict[str, float]]:
        file_path = self.get_data_path(dataset_path, info)

        data = load_json(file_path)
        return data




info_getter_builders: dict[str, Callable[[], InfoGetter]] = {
    "pandaset": PandasetInfoGetter,
    "neurad": NeuRADInfoGetter,
}

data_getter_builders: dict[str, Callable[[InfoGetter, dict[str, Any]], DataGetter]] = {
    "rgb": RGBDataGetter,
    "meta": MetaDataGetter,
    "lidar": LidarDataGetter,
    "prompt": PromptDataGetter,
    "intrinsics": IntrinsicsDataGetter,
    "pose": PoseDataGetter
}


class DynamicDataset(Dataset):  # Dataset / Scene / Sample
    def __init__(
        self,
        dataset_path: Path,
        data_tree: dict[str, Any],
        info_getter: InfoGetter,
        data_getters: dict[str, DataGetter],
        data_transforms: dict[str, Callable[[str], int]] = None,
        preprocess_func = None
    ):
        self.sample_infos: list[SampleInfo] = info_getter.parse_tree(
            dataset_path, data_tree
        )
        self.info_getter = info_getter
        self.data_getters = data_getters
        self.dataset_path = dataset_path
        self.preprocess_func = preprocess_func

        self.data_transforms = {**data_transforms} if data_transforms else {}
        for data_type in data_getters.keys():
            if data_type not in self.data_transforms:
                self.data_transforms[data_type] = []

        self.reset_index()

    def reset_index(self):
        self.idxs = np.arange(len(self.sample_infos))

    def shuffle_index(self):
        np.random.shuffle(self.idxs)

    def limit_size(self, size: float | int):
        if size is None:
            size = self.true_len

        elif isinstance(size, float):
            size = int(self.true_len * size)

        self.idxs = self.idxs[:size]

    @property
    def true_len(self):
        return len(self.sample_infos)

    @classmethod
    def from_config(cls, dataset_config: Path | dict[str, Any]):
        if isinstance(dataset_config, Path):
            dataset_config = read_yaml(dataset_config)

        dataset_path = Path(dataset_config["path"])
        data_tree = read_data_tree(dataset_config["data_tree"])
        dataset_name = dataset_config["dataset"]

        info_getter_factory = info_getter_builders[dataset_name]
        info_getter = info_getter_factory()

        data_getters = {}

        for data_type, spec in dataset_config["data_getters"].items():
            if data_type not in data_getter_builders:
                raise NotImplementedError(f"Could not find a builder for {data_type}")

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
        """ Iterate cyclically with a range, such that a specific offset is matched with the right index.
            Example: (id: 14, id_start: 10, id_stop: 25)
                id=10 should be assigned i=0, id=11 gets i=1, etc...
                This continues until id=25, after which it repeats at id=10.
                We therefore get the map: 
                {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, ..., 
                 10: 15, 11: 16, 12: 17, 13: 18, 14: 19, ...}
                Thus, id=14 will be assigned indexes 4, 19, 34, 49, ..., until the dataset is exhausted.

        Args:
            id: Offset between the id range and index range. Defaults to 0.
            id_start: Starting index of the cycle. Defaults to 0.
            id_stop (int, optional): Final id in the index range before restarting the cycle. Defaults to 0.
            verbose: Whether or not to log each index. Defaults to True.

        Yields:
            Sample at specified index.
        """
        assert id_start <= id <= id_stop

        skip = id_stop - id_start + 1
        start = id - id_start
        stop = len(self)

        for i in range(start, stop, skip):
            if verbose:
                logging.info(f"Sample {i}/{stop} (skip: {skip})")

            yield self[i]

    def __len__(self) -> int:
        return len(self.idxs)

    def __iter__(self) -> Generator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = self.idxs[idx]

        info = self.sample_infos[i]
        sample = {}

        for data_type, getter in self.data_getters.items():
            data = getter.get_data(self.dataset_path, info)
            
            for data_transform in self.data_transforms[data_type]:
                data = data_transform(data)

            sample[data_type] = data

        if self.preprocess_func:
            sample = self.preprocess_func(sample)

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
