#!/bin/bash
# SBATCH --partition=public-gpu
# SBATCH --time=2-00:00:00
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
# SBATCH --mem-per-gpu=90000 # in MB
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=./Results/logs2.log


echo $SLURM_JOBID

module load Anaconda3
source /opt/ebsofts/Anaconda3/2021.05/etc/profile.d/conda.sh


# conda activate DeepFocus_env


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -u FinalDeepFocus2.py
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -u FinalDeepFocus_resnet.py

