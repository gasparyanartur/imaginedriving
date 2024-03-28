import argparse
from pathlib import Path

from src.data import DirectoryDataset, NamedImageDataset, DynamicDataset
from src.data import save_json, read_data_tree, get_dataset_from_path
from src.benchmark import benchmark_single_metrics, benchmark_aggregate_metrics
from src.benchmark import run_benchmark_suite
from src.configuration import read_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_benchmarks")

    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    config_path = args.config_path
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir

    config = read_yaml(config_path)
    data_tree = read_data_tree(config)

    pred_dataset_name = get_dataset_from_path(pred_dir)
    gt_dataset_name = get_dataset_from_path(gt_dir)
    

    pred_dataset = DynamicDataset(pred_dir, data_tree, )

    pred_dataset = DirectoryDataset.from_directory(args.pred_dir) 
    gt_dataset = DirectoryDataset.from_directory(args.gt_dir)

    single_metrics = benchmark_single_metrics(pred_dataset, gt_dataset)
    aggregate_metrics = benchmark_aggregate_metrics(pred_dataset, gt_dataset)

    args.save_dir.mkdir(exist_ok=True, parents=True)

    save_json(metrics) 