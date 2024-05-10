
wandb_api_key=${WANDB_API_KEY}
if [ -z ${wandb_api_key} ]; then
    echo "WANDB_API_KEY not set. Exiting."
    exit 1;
fi

image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
config_path=${CONFIG_PATH:-"configs/hal-configs/train_model.yml"}
workdir=${WORKDIR:-"/home/s0001900/workspace/imaginedriving"}

singularity exec --env PYTHONPATH=$workdir --env WANDB_API_KEY=$wandb_api_key --bind /staging:/staging -H $workdir --nv $image_path python -m pdb scripts/run_train_lora.py $config_path