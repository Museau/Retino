#!/bin/bash
#SBATCH --time=0:30:0
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G


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

# Preprocess the labels
for split in train valid test; do
    time python generate_json_dataset_splits.py --data_folder $SCRATCH/kaggle2015 --split $split;
done
