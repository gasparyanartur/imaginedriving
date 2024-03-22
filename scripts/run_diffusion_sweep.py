#import sys; sys.path.insert(0, 'C:\\Users\\gaspa\\dev\\projects\\nerf-thesis')

import logging
from argparse import ArgumentParser
from pathlib import Path
import yaml


from src.configuration import setup_project, read_yaml, save_yaml
from src.diffusion import ImgToImgModel
from src.utils import get_parameter_combinations, combine_kwargs, show_img
from src.data import read_image, save_image
from src.benchmark import DefaultMetricWrapper


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("img_path", type=Path)
    parser.add_argument("experiment_config_path", type=Path)
    parser.add_argument("-d", "--experiment_dir", type=Path, default=None)
    parser.add_argument("-s", "--show_img", action="store_true")

    args = parser.parse_args()

    img_path = args.img_path
    experiment_config_path = args.experiment_config_path
    experiment_dir = args.experiment_dir
    flag_show_img = args.show_img


    if experiment_dir is None and flag_show_img is False:
        raise ValueError(f"Need to either show image or store result")

    setup_project()

    if not img_path.exists():
        raise ValueError(f"Could not find img_path: {img_path}")

    if not experiment_config_path.exists():
        raise ValueError(f"Could not find experiment_config_path: {experiment_config_path}")

    config = read_yaml(experiment_config_path) 

    img = read_image(img_path)
    img_name = img_path.stem

    logging.info("Running experiments...")

    prev_model_config_params = None
    model: ImgToImgModel = None

    if experiment_dir:
        experiment_dir.mkdir(exist_ok=True, parents=True)

    for exp_name, experiment in config["experiments"].items():
        model_config_params = experiment["model_config_params"]
        if model_config_params != prev_model_config_params:
            prev_model_config_params = model_config_params
            model = ImgToImgModel.load_model(model_config_params)

        model_forward_params = experiment["model_forward_params"]
        diffused_img = model.img_to_img(img, **model_forward_params)["image"]

        # TODO: Add benchmark

        if flag_show_img:
            show_img((img, diffused_img))

        if experiment_dir:
            experiment_dir = experiment_dir / exp_name
            experiment_dir.mkdir(exist_ok=True)

            save_yaml(experiment_dir / "config.yml", experiment)
            save_image(experiment_dir / img_name, diffused_img)
