#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/nlp4adas/master-thesis-shared/logs/benchmark/%A_%a.out
#SBATCH -A Berzelius-2024-1
#SBATCH --job-name=benchmark

image_path=${IMAGE_PATH:-"/proj/nlp4adas/containers/nerf-thesis-0.4.sif/"}
config_dir=${CONFIG_DIR:-"configs/benchmark-pandaset-real_neurad.yml"}

singularity exec --nv $image_path python scripts/run_benchmarks.py $config_dir