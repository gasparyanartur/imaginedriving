from pathlib import Path

from src.diffusion import ImageToImageDiffusionModel
from src.configuration import setup_project
from src.data import load_img_paths_from_dir, read_image
from src.utils import show_img


if __name__ == "__main__":
    config = setup_project(config_path=None)
    """
    dataset_config = config["datasets"]["pandaset"]
    dataset_path = Path(dataset_config["path"])
    data_path = dataset_path / dataset_config["scenes"][0] / "camera" / "front_camera"
    data_path = Path("/home/s0001900/Pictures/Renders/renders/shift-0m")
    img_paths = load_img_paths_from_dir(data_path)
    ex_img_path = img_paths[0]
    """
    ex_img_path = Path("/home/s0001900/Pictures/Renders/renders/shift-0m/09.jpg")
    ex_img = read_image(ex_img_path)

    pipe = ImageToImageDiffusionModel(low_mem_mode=True)

    breakpoint()
    img = pipe.forward(ex_img)
    show_img(img)