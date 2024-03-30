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
source_dir=${SOURCE_DIR:-"/proj/adas-data/data/pandaset"}
dest_dir=${DEST_DIR:-"/proj/nlp4adas/master-thesis-shared/output/param_sweep"}
config_dir=${CONFIG_DIR:-"configs/paramsweep-config.yml"}


if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    echo "SLURM_ARRAY_TASK_ID variable is empty. Exiting." ;
    exit 1;
fi


config_array=($(find $configs_dir -name '*.yml'))
n_paths=${#config_array[@]}

if [ $n_paths -gt $array_end ]; then
    echo "Too many configurations found in $config_dir. Expected $array_end, found $n_paths. Exiting." ;
    exit 1;
fi

if [ $SLURM_ARRAY_TASK_ID -gt $array_end ]; then
    echo "Array Task ID $SLURM_ARRAY_TASK_ID exceeds number of paths $n_paths. Exiting." ;
    exit 0;
fi

singularity exec --nv $image_path python scripts/run_diffusion_sweep.py $source_dir $dest_dir $config_dir -id $SLURM_ARRAY_TASK_ID $array_start $array_end