from pathlib import Path
from diffusers.utils import make_image_grid    

from src.diffusion import ImageToImageDiffusionModel
from src.configuration import setup_project
from src.data import load_data_from_path, read_image_from_sample

config = setup_project(config_path=None)
dataset_config = config["datasets"]["pandaset"]
dataset_path = Path(dataset_config["path"])
data_path = dataset_path / dataset_config["scenes"][0] / "camera" / "front_camera"

data = load_data_from_path(data_path)
pipe = ImageToImageDiffusionModel()
ex_img = read_image_from_sample(data, 0)
img = pipe.forward(ex_img)


def main():
    ...


if __name__ == "__main__":
    main()