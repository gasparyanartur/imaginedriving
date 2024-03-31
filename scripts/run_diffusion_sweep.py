from argparse import ArgumentParser
from pathlib import Path

from src.data import setup_project
from src.diffusion import diffusion_from_config_to_dir
from src.data import DynamicDataset, read_yaml


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("-id", "--id_range", nargs=3, type=int, default=None)

    args = parser.parse_args()
    config_path = args.config_path
    id_range = args.id_range

    setup_project()

    config = read_yaml(config_path)
    dataset_config = config["datasets"]["source_images"]
    src_dataset = DynamicDataset.from_config(dataset_config)
    output_path = Path(config["output_dir"])

    for experiment_name, model_config in config["experiments"].items():
        diffusion_from_config_to_dir(src_dataset, output_path / experiment_name, model_config, id_range=id_range)
    