import argparse
from pathlib import Path

from src.benchmark import Metrics, save_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_benchmarks")

    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument("--print", action="store_true")

    args = parser.parse_args()

    metrics = Metrics.get_default()
    results = metrics.benchmark_dir_to_dir(args.pred_dir, args.gt_dir)
    save_metrics(results, args.save_path, print_results=args.print)