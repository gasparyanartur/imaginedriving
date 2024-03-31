#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/nlp4adas/master-thesis-shared/logs/benchmark/%A_%a.out
#SBATCH -A Berzelius-2024-1
#SBATCH --job-name=benchmark
#SBATCH --array=0-10
max_arrays=10

if [ -z ${PRED_DIR_ARRAY} ]; then
    echo "PRED_DIR_ARRAY variable is empty. Exiting."
    exit 1;
fi

if [ -z ${GT_DIR} ]; then
    echo "GIT_DIR variable is empty. Exiting."
    exit 1;
fi

if [ -z ${EXPERIMENT_DIR} ]; then
    echo "EXPERIMENT_DIR variable is empty. Exiting."
    exit 1;
fi

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    echo "SLURM_ARRAY_TASK_ID variable is empty. Exiting." ;
    exit 1;
fi

readarray pred_dirs < "scripts/benchmark-dir-array.txt"
n_pred_dirs=${#pred_dirs[@]}

if [ "$n_pred_dirs" -gt ${max_arrays} ]; then
    echo "Too many pred_dirs found in ${pred_dirs}. Expected ${max_arrays}, found ${n_src_dirs}. Exiting." ;
    exit 1;
fi

if [ "$SLURM_ARRAY_TASK_ID" -gt ${n_pred_dirs} ]; then
    echo "Array Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of jobs ${n_pred_dirs}. Exiting." ;
    exit 0;
fi

pred_dir=${pred_dirs[${SLURM_ARRAY_TASK_ID}]}
singularity exec --nv ~/base/containers/nerf-thesis-latest.sif/ python scripts/run_benchmarks.py ${pred_dir} ${GT_DIR} --save_dir ${EXPERIMENT_DIR}