#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --J env_creation
#SBATCH --N 1
#SBATCH --n 1
#SBATCH --p vm-small
#SBATCH --t 00:30:00
#SBATCH --A MLL

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_$current_date

# Create the conda environment

source /work/09753/hprairie/ls6/miniconda3/etc/profile.d/conda.sh
conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"


