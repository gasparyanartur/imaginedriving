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

