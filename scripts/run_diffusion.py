import logging
from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project
from src.diffusion import ModelId, ImageToImageDiffusionModel, diffuse_images_to_dir
from src.data import load_img_paths_from_dir


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("images_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--model_id", type=str, default=ModelId.sdxl_ref_v1_0)

    args = parser.parse_args()

    logging.info("Setting up project...")
    config = setup_project()
    logging.info("Finished setting up project")

    logging.info(f"Loading diffusion model: args.model_id")
    model = ImageToImageDiffusionModel(args.model_id)
    logging.info(f"Finished loading diffusion model")

    logging.info(f"Diffusing images from directory {args.images_dir} to directory {args.dst_dir}")
    img_paths = load_img_paths_from_dir(args.images_dir)
    diffuse_images_to_dir(model, img_paths, dst_dir=args.dst_dir)
    logging.info(f"Finished diffusion, exiting program")
