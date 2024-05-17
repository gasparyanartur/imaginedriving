#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task 8
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/finetune/slurm/%j.out
#SBATCH --partition=zprodlow 
#SBATCH --job-name=finetune

main_process_port=${MAIN_PROCESS_PORT:-29500}
num_processes=${NUM_PROCESSES:-1}
num_machines=${NUM_MACHINES:-1}
dynamo_backend=${DYNAMO_BACKEND:-"no"}
mixed_precision=${MIXED_PRECISION:-"no"}

wandb_api_key=${WANDB_API_KEY}
if [ -z ${wandb_api_key} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
config_path=${CONFIG_PATH:-"configs/hal-configs/train_model.yml"}
workdir=${WORKDIR:-"/home/s0001900/workspace/imaginedriving"}

singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path accelerate launch --num_processes=$num_processes --num_machines=$num_machines --dynamo_backend=$dynamo_backend --mixed_precision=$mixed_precision --main_process_port=$main_process_port scripts/run_train_lora.py $config_path