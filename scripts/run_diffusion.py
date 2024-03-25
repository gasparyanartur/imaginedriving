from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project, read_yaml
from src.diffusion import diffusion_from_config_to_dir
from src.data import DirectoryDataset
from src.utils import get_device


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("-m", "--model_config_path", type=Path, default=None)
    parser.add_argument("-sb", "--skip_benchmark", action="store_true")

    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    model_config_path = args.model_config_path
    skip_benchmark = args.skip_benchmark

    setup_project()
    device = get_device()

    if not src_dir.exists():
        raise ValueError(f"Could not find source dir: {src_dir}")

    src_dataset = DirectoryDataset.from_directory(src_dir, device=device)
    model_config = read_yaml(model_config_path) if model_config_path else None

    diffusion_from_config_to_dir(src_dataset, dst_dir, model_config, device)
    
