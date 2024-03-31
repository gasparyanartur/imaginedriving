import argparse
from pathlib import Path

from src.data import save_json, read_data_tree, DynamicDataset, setup_project
from src.benchmark import run_benchmark_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_benchmarks")

    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    config_path = args.config_path

    config = setup_project(config_path)

    dataset_config = config["datasets"]
    preds_config = dataset_config["preds"]
    gts_config = dataset_config["gts"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    preds_dataset = DynamicDataset.from_config(preds_config)
    gts_dataset = DynamicDataset.from_config(gts_config)

    metrics = run_benchmark_suite(preds_dataset, gts_dataset)
    save_json(output_dir / "metrics.json", metrics) 