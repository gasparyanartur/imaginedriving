import logging
from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project
from src.diffusion import ModelId, diffuse_images_to_dir, ImgToImgModel
from src.data import load_img_paths_from_dir, DirectoryDataset


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--model_config_path", type=Path, default=None)

    args = parser.parse_args()

    if not args.src_dir.exists():
        raise ValueError(f"Could not find source dir: {args.src_dir}")

    src_dataset = DirectoryDataset.from_directory(args.src_dir)
    dst_dataset = DirectoryDataset.from_directory(args.dst_dir)

    logging.info("Setting up project...")
    config = setup_project()
    logging.info("Finished setting up project")

    logging.info(f"Loading diffusion model: {args.model_id}")
    model = ImgToImgModel()
    logging.info(f"Finished loading diffusion model")

    logging.info(f"Diffusing images from directory {args.src_dir} to directory {args.dst_dir}")
    img_paths = load_img_paths_from_dir(args.src_dir)
    diffuse_images_to_dir(model, img_paths, dst_dir=args.dst_dir)
    logging.info(f"Finished diffusion, exiting program")
