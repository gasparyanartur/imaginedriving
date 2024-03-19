#import sys; sys.path.insert(0, 'C:\\Users\\gaspa\\dev\\projects\\nerf-thesis')

import logging
from argparse import ArgumentParser
from pathlib import Path
import yaml


from src.configuration import setup_project, read_yaml
from src.diffusion import SDXLFull


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("img_path", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("experiment_config_path", type=Path)

    args = parser.parse_args()
    setup_project()

    if not args.img_path.exists():
        raise ValueError(f"Could not find img_path: {args.img_path}")

    if not args.experiment_config_path.exists():
        raise ValueError(f"Could not find experiment_config_path: {args.experiment_config_path}")

    logging.info(f"Loading diffusion model...")
    #model = SDXLFull()
    logging.info(f"Finished loading diffusion model")

    config = read_yaml(args.experiment_config_path) 
    args.experiment_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Running experiments...")
    for exp_name, experiment in config["experiments"].items():
        print(exp_name)
