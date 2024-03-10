#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/nlp4adas/users/%u/logs/%A_%a.out
#SBATCH -A Berzelius-2024-1
#SBATCH --array=1-10
#SBATCH --job-name=ours
#


if [ -z ${SOURCE_DIR} ]; then
    echo "SOURCE_DIR variable is empty. Exiting." ;
    exit 1;
fi

if [ -z ${DEST_DIR} ]; then
    echo "DEST_DIR variable is empty. Exiting." ;
    exit 1;
fi



# Specify the path to the config file
id_to_seq=scripts/arrays/${dataset}_id_to_seq${SUFFIX}.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
[[ -z $seq ]] && exit 1

# For each sequence, start the training
echo "Starting training for $name with extra args ${@:2}"

echo "Sequence $seq"

output_dir=/proj/nlp4adas/master-thesis-shared/radnerf/$dataset-$method
mkdir -p $output_dir

if [ -z ${LOAD_NAME+x} ]; then
    MAYBE_RESUME_CMD=""
else
    echo "LOAD_NAME specified in environment, resuming from $LOAD_NAME"
    checkpoints=( $(ls outputs/$LOAD_NAME-$seq/$method/*/nerfstudio_models/*.ckpt) )
    MAYBE_RESUME_CMD="--load-checkpoint=${checkpoints[-1]}"
fi

singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /proj:/proj \
    --pwd /nerfstudio \
    /proj/nlp4adas/containers/nerfstudio_20231019.sif \
    python -u nerfstudio/scripts/train.py \
    $method \
    --output-dir $output_dir \
    --vis wandb \
    --experiment-name $name-$seq \
    $MAYBE_RESUME_CMD \
    ${@:2} \
    ${dataset}-data \
    --data /proj/adas-data/data/${dataset} \
    --sequence $seq \
    $DATAPARSER_ARGS

#
#EOF
