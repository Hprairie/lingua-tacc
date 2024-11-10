#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name data_download
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p vm-small
#SBATCH -t 01:00:00
#SBATCH -A MLL

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_241028
tokenizer=cl100k_base
directory=$SCRATCH/tokenizers/gpt2

# Enable the conda environment
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $env_prefix

echo "Currently in env $(which python)"

# Download the dataset
cd $WORK/projects/lingua-tacc
python setup/download_tokenizer.py --tokenizer-name $tokenizer --tokenizer-dir $directory

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))
echo "Tokenizer $tokenizer created at $directory and installed successfully in $elapsed_minutes minutes!"