import argparse
from pathlib import Path

from src.data import DirectoryDataset
from src.benchmark import benchmark_single_metrics, benchmark_aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_benchmarks")

    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--silent", action="store_true")

    args = parser.parse_args()

    pred_dataset = DirectoryDataset.from_directory(args.pred_dir) 
    gt_dataset = DirectoryDataset.from_directory(args.gt_dir)

    single_metrics = benchmark_single_metrics(pred_dataset, gt_dataset)
    aggregate_metrics = benchmark_aggregate_metrics(pred_dataset, gt_dataset)

    if not args.silent:
        print()
        print(f"Running benchmarks:")
        print(f"\tpred: {args.pred_dir}")
        print(f"\tgt: {args.gt_dir}")
        print()
        print("========================")
        print("===== Sin. Metrics =====")
        print("========================")
        print(single_metrics)
        print()
        print("========================")
        print("===== Agg. Metrics =====")
        print("========================")
        print(aggregate_metrics)
        print()

    if args.save_dir:
        args.save_dir.mkdir(exist_ok=True, parents=True)

        single_metrics.to_csv(args.save_dir / "single_metrics.csv") 
        aggregate_metrics.to_csv(args.save_dir / "aggregate_metrics.csv") 