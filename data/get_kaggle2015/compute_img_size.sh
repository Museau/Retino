#!/bin/bash
#SBATCH --time=2:00:0
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G


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

# Copy data to compute node
mkdir $SLURM_TMPDIR/kaggle2015

# Preprocess the images
# Unzip the files
7z x $SCRATCH/kaggle2015/diabetic-retinopathy-detection.zip -o$SLURM_TMPDIR/kaggle2015
7z x $SLURM_TMPDIR/kaggle2015/train.zip.001 -o$SLURM_TMPDIR/kaggle2015
7z x $SLURM_TMPDIR/kaggle2015/test.zip.001 -o$SLURM_TMPDIR/kaggle2015

# Compute and plot image size
python -u ../compute_img_size.py kaggle_train $SLURM_TMPDIR/kaggle2015/train
python -u ../compute_img_size.py kaggle_test $SLURM_TMPDIR/kaggle2015/test
