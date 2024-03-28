import logging
from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project, read_yaml, save_yaml
from src.diffusion import DiffusionModel
from src.utils import show_img, get_device
from src.data import read_image, save_image, NamedImageDataset, DirectoryDataset
from src.benchmark import benchmark_single_metrics


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("experiment_config_path", type=Path)
    parser.add_argument("base_experiment_dir", type=Path)
    parser.add_argument("-s", "--show_img", action="store_true")

    args = parser.parse_args()

    src_dir = args.src_dir
    experiment_config_path = args.experiment_config_path
    base_experiment_dir = args.base_experiment_dir
    flag_show_img = args.show_img

    setup_project()

    if not src_dir.exists():
        raise ValueError(f"Could not find src_dir: {src_dir}")

    if not experiment_config_path.exists():
        raise ValueError(f"Could not find experiment_config_path: {experiment_config_path}")

    config = read_yaml(experiment_config_path) 
    src_ds = DirectoryDataset.from_directory(src_dir)

    logging.info("Running experiments...")
    prev_model_config_params = None
    model: DiffusionModel = None
    base_experiment_dir.mkdir(exist_ok=True, parents=True)

    device = get_device()
    img_name = img_path.stem
    img = read_image(img_path)
    img = img.to(device)

    for exp_name, experiment in config["experiments"].items():
        logging.info(f"Running experiment {exp_name} with parameters: {experiment}")

        model_config_params = experiment["model_config_params"]
        if model_config_params != prev_model_config_params:
            prev_model_config_params = model_config_params
            model = DiffusionModel.load_model(model_config_params)

        model_forward_params = experiment["model_forward_params"]
        diffused_img = model.diffuse_sample(img, **model_forward_params)["rgb"]

        if flag_show_img:
            show_img((img, diffused_img))

        experiment_dir = base_experiment_dir / exp_name
        experiment_dir.mkdir(exist_ok=True)
        diff_img_path = (experiment_dir / img_name).with_suffix(".jpg")

        logging.info(f"Saving experiment to {experiment_dir}")
        save_yaml(experiment_dir / "config.yml", experiment)
        save_image(diff_img_path, diffused_img)

        logging.info(f"Finished running experiment")

