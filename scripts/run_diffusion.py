import logging
from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project, read_yaml
from src.diffusion import load_img2img_model
from src.data import load_img_paths_from_dir, DirectoryDataset


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--model_config_path", type=Path, default=None)

    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    model_config_path = args.model_config_path

    setup_project()

    if not src_dir.exists():
        raise ValueError(f"Could not find source dir: {src_dir}")

    src_dataset = DirectoryDataset.from_directory(src_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)
    model_config = read_yaml(model_config_path) if model_config_path else None

    logging.info(f"Loading diffusion model with config: {model_config}")
    model = load_img2img_model(model_config_path)
    logging.info(f"Finished loading diffusion model")

    logging.info(f"Diffusing images from directory {src_dir} to directory {args.dst_dir}")
    img_paths = load_img_paths_from_dir(src_dir)
    model.src_dataset_to_dir(src_dataset, dst_dir)
    logging.info(f"Finished diffusion, exiting")
