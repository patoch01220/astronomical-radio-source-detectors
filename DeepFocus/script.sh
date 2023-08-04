#!/bin/bash
# SBATCH --partition=public-gpu
# SBATCH --time=2-00:00:00
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=./Results/logs.log


echo $SLURM_JOBID

module load Anaconda3
source /opt/ebsofts/Anaconda3/2021.05/etc/profile.d/conda.sh


conda activate DeepFocus_final
# conda activate test
# conda uninstall pytorch torchvision cudatoolkit -c pytorch
# conda install pytorch torchvision cudatoolkit -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install torch
# conda install matplotlib 
# conda install astropy

# python3 -u FinalDeepFocus.py
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -u FinalDeepFocus.py

# conda update -n base -c defaults conda