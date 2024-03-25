#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/nlp4adas/master-thesis-shared/logs/param-sweep/%A_%a.out
#SBATCH -A Berzelius-2024-1
#SBATCH --job-name=param-sweep
#SBATCH --array=0-10
max_config_files="10"       # Don't exceed the number of arrays

if [ -z ${SOURCE_DIR} ]; then
    echo "SOURCE_DIR variable is empty. Exiting." ;
    exit 1;
fi

if [ -z ${DEST_DIR} ]; then
    echo "DEST_DIR variable is empty. Exiting." ;
    exit 1;
fi

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    echo "SLURM_ARRAY_TASK_ID variable is empty. Exiting." ;
    exit 1;
fi

if [ -z ${CONFIGS_DIR} ]; then
    echo "CONFIGS_DIR variable is empty. Exiting." ;
    exit 1;
fi

config_array=($(find ${CONFIGS_DIR} -name '*.yml'))
n_paths=${#config_array[@]}

if [ "$n_paths" -gt ${max_config_files} ]; then
    echo "Too many configurations found in ${CONFIGS_DIR}. Expected ${max_config_files}, found ${n_paths}. Exiting." ;
    exit 1;
fi

if [ "$SLURM_ARRAY_TASK_ID" -gt ${max_config_files} ]; then
    echo "Array Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of paths ${n_paths}. Exiting." ;
    exit 0;
fi

config_path=${config_array[$SLURM_ARRAY_TASK_ID]}
config_name=$(basename -s ".yml" ${config_path})

singularity exec --nv ~/base/containers/nerf-thesis-latest.sif/ python scripts/run_diffusion.py ${SOURCE_DIR} ${DEST_DIR}/${config_name} -m ${config_path}