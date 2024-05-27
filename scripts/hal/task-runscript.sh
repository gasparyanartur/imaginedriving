#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/logs/generic/slurm/%j.out
#SBATCH --partition=zprodlow 
#SBATCH --job-name=generic

image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/nerf-thesis.sif"}
singularity exec  --env PYTHONPATH=$PWD --bind /staging:/staging -H $PWD --nv $image_path ${@:1}