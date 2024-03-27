from argparse import ArgumentParser
from pathlib import Path

from src.configuration import setup_project, read_yaml
from src.diffusion import diffusion_from_config_to_dir
from src.data import DynamicDataset, read_data_tree, PandasetInfoGetter, PandasetImageDataGetter
from src.utils import get_device


if __name__ == "__main__":
    parser = ArgumentParser("diffusion")

    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    config_path = args.config_path

    setup_project()

    if not dataset_path.exists():
        raise ValueError(f"Could not find dataset_path: {dataset_path}")

    config = read_yaml(config_path)
    data_tree = read_data_tree(config["data_tree"])

    info_getter = PandasetInfoGetter()
    data_getters = {"image": PandasetImageDataGetter()}
    src_dataset = DynamicDataset(dataset_path, data_tree, info_getter, data_getters)

    diffusion_from_config_to_dir(src_dataset, output_path, config)
    
