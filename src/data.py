from pathlib import Path
import json
import PIL
from PIL import Image


def sort_paths_numerically(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: int(path.stem))


def load_json(path):
    with open(path, "rb") as f:
        return json.load(f)


def load_img_paths_from_dir(dir_path: Path):
    img_paths = list(sorted(dir_path.glob("*.jpg")))
    return img_paths


def read_PIL_img(img_path):
    with PIL.Image.open(img_path) as img:
        return img.convert("RGB")


def save_PIL_img(save_path: Path, img: Image):
    img.save(save_path)


def read_image_from_sample(data, sample_idx):
    img_path = data["img_paths"][sample_idx]
    return read_PIL_img(img_path)
