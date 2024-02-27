from pathlib import Path

from src.diffusion import ImageToImageDiffusionModel
from src.configuration import setup_project
from src.data import load_img_paths_from_dir, read_image

config = setup_project(config_path=None)
dataset_config = config["datasets"]["pandaset"]
dataset_path = Path(dataset_config["path"])
data_path = dataset_path / dataset_config["scenes"][0] / "camera" / "front_camera"

img_paths = load_img_paths_from_dir(data_path)
ex_img = read_image(img_paths[0])

pipe = ImageToImageDiffusionModel()
