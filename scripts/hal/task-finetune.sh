#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/finetune/%j.out
#SBATCH --partition=zprodlow 
#SBATCH --job-name=finetune

wandb_api_key=${WANDB_API_KEY}
if [ -z ${wandb_api_key} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
config_path=${CONFIG_PATH:-"configs/hal-configs/train_model.yml"}
workdir=${WORKDIR:-"/home/s0001900/workspace/nerf-thesis"}

singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --main_process_port 0 scripts/run_train_model_lora.py $config_path