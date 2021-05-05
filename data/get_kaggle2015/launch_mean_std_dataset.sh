#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G

# Those cpu and mem requirement are for a full node on beluga.
# Note: If you don't have access to enough memory use flag `--low-mem`.

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

mkdir $SLURM_TMPDIR/kaggle2015

## Compute mean and std
# For train 512
tar -C $SLURM_TMPDIR/kaggle2015 -xf $SCRATCH/kaggle2015/train_512.tar
time python -u ../compute_image_set_stats.py $SLURM_TMPDIR/kaggle2015/train_512

# For train 1024
tar -C $SLURM_TMPDIR/kaggle2015 -xf $SCRATCH/kaggle2015/train_1024.tars
time python -u ../compute_image_set_stats.py $SLURM_TMPDIR/kaggle2015/train_1024
