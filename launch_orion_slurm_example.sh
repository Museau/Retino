#!/bin/bash
#SBATCH --array=1-40
#SBATCH --time=24:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/<user>/scratch/logs/slurm/exp.%A.%a.out
#SBATCH --error=/home/<user>/scratch/logs/slurm/exp.%A.%a.err

# NOTE : /home/<user>/scratch/logs/slurm/ needs to be created in advance
# NOTE : ^ Orion does not like when the logs are saved in the repository

## Export path to mlflow folder
export MLFLOW_TRACKING_URI=$SCRATCH/logs/mlruns

## Allows messages to be immediately dumped to the stream instead of being buffered
export PYTHONUNBUFFERED=TRUE

## Experiment variable
EXP_NAME=kaggle_cnn4layers2conv_bn
IMG_SIZE=512

## Setup log folder
LOG_FOLDER=$SCRATCH/logs/$EXP_NAME
mkdir -p $LOG_FOLDER
LOG_FILE=$LOG_FOLDER/exp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

## Prep data to local node SSD
mkdir $SLURM_TMPDIR/kaggle2015
tar -C $SLURM_TMPDIR/kaggle2015 -xf $SCRATCH/kaggle2015/train_$IMG_SIZE.tar
tar -C $SLURM_TMPDIR/kaggle2015 -xf $SCRATCH/kaggle2015/test_$IMG_SIZE.tar
cp $SCRATCH/kaggle2015/labels_train.json $SLURM_TMPDIR/kaggle2015
cp $SCRATCH/kaggle2015/labels_valid.json $SLURM_TMPDIR/kaggle2015

## Unpack and load env
# Make sure no previous conda env is activated
if command -v conda; then
  conda deactivate;
fi

# Load packed conda env
ENV_PATH=$SLURM_TMPDIR/retino
mkdir -p $ENV_PATH
tar -xzf $HOME/retino.tar.gz -C $ENV_PATH
source $ENV_PATH/bin/activate
conda-unpack


## Hack to not start jobs at the same time (wait N sec) to minimize the risk of writing to the database at the same time
sleep $(($SLURM_ARRAY_TASK_ID * 2))

## Start Experiment
orion -vvv hunt -c .orionconfig.yml -n $EXP_NAME --worker-max-trials 1 python main.py --data-folder $SLURM_TMPDIR/kaggle2015/ --experiment-folder $LOG_FOLDER/checkpoint --config orion.example.conf.yml &> $LOG_FILE
