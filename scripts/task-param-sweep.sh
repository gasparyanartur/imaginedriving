#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/nlp4adas/master-thesis-shared/logs/param-sweep/%A_%a.out
#SBATCH -A Berzelius-2024-1
#SBATCH --job-name=param-sweep
#SBATCH --array=0-10
array_start=0
array_end=10

image_path=${IMAGE_PATH:-"/proj/nlp4adas/containers/nerf-thesis-0.4.sif/"}
config_dir=${CONFIG_DIR:-"configs/param_sweep.yml"}

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    echo "SLURM_ARRAY_TASK_ID variable is empty. Exiting." ;
    exit 1;
fi

singularity exec --nv $image_path python scripts/run_diffusion_sweep.py $config_dir -id $SLURM_ARRAY_TASK_ID $array_start $array_end