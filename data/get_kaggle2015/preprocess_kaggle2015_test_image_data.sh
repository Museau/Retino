#!/bin/bash
#SBATCH --time=6:0:0
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G


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

for size in 512 1024; do
    mkdir $SLURM_TMPDIR/kaggle2015/test_$size;
done

# Preprocess the images
# Unzip the files
7z x $SCRATCH/kaggle2015/diabetic-retinopathy-detection.zip -o$SLURM_TMPDIR/kaggle2015
7z x $SLURM_TMPDIR/kaggle2015/test.zip.001 -o$SLURM_TMPDIR/kaggle2015

# Crop and resize the images into 512x512 1024x1024
time python -u crop_resize_kaggle2015_image_data.py $SLURM_TMPDIR/kaggle2015/test

cd $SLURM_TMPDIR/kaggle2015/

# Compress the data sets
for size in 512 1024; do
    tar -cvf $SCRATCH/kaggle2015/test_$size.tar test_$size;
done
