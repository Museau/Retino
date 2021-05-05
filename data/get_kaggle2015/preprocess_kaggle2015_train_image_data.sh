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
    mkdir $SLURM_TMPDIR/kaggle2015/train_$size;
done

# Preprocess the images
# Unzip the files
7z x $SCRATCH/kaggle2015/diabetic-retinopathy-detection.zip -o$SLURM_TMPDIR/kaggle2015
7z x $SLURM_TMPDIR/kaggle2015/trainLabels.csv.zip -o$SCRATCH/kaggle2015 -aot
7z x $SLURM_TMPDIR/kaggle2015/train.zip.001 -o$SLURM_TMPDIR/kaggle2015

# Crop and resize the images into 512x512 1024x1024
time python -u crop_resize_kaggle2015_image_data.py $SLURM_TMPDIR/kaggle2015/train

cd $SLURM_TMPDIR/kaggle2015/
# Removing patients that have low quality images such that automatic preprocessing failed
for size in 512 1024; do
    for patient_id in 492 1986 32253 34689 43457; do
        rm train_${size}/${patient_id}_right.jpeg;
        rm train_${size}/${patient_id}_left.jpeg;
    done;
done

# Compress the data sets
for size in 512 1024; do
    tar -cvf $SCRATCH/kaggle2015/train_$size.tar train_$size;
done
