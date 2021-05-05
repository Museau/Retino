#!/bin/bash


# This script expect that you have kaggle installed and accessible

# To use the Kaggle API, you need to create a 'API token'
# https://github.com/Kaggle/kaggle-api
export KAGGLE_USERNAME=<kaggle_username>
export KAGGLE_KEY=<kaggle_key>

mkdir $SCRATCH/kaggle2015
cd $SCRATCH/kaggle2015

# Get train/test images and train labels
kaggle competitions download -c diabetic-retinopathy-detection

# Get test labels
wget https://storage.googleapis.com/kaggle-forum-message-attachments/90528/2877/retinopathy_solution.csv

# Get train/test quality labels
wget https://raw.githubusercontent.com/HzFu/EyeQ/master/data/Label_EyeQ_train.csv
wget https://raw.githubusercontent.com/HzFu/EyeQ/master/data/Label_EyeQ_test.csv
