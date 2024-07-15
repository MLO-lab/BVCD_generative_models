#!/bin/bash
#SBATCH --job-name=MNIST-DDPM # name of the job
#SBATCH -p gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./slurm_logs/MNIST-DDPM-%j.out
#SBATCH --error=./slurm_logs/MNIST-DDPM-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
##SBATCH -w gpu[29,30,31,33]
#SBATCH --mem=6000
##SBATCH -N 5
##SBATCH -a 0-120%3

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

## train single diffusion model
python3 mnist__conditional_diffusion.py --set_id $1 --n_epoch $2 --zero_frac $3 --model_folder $4 --deep_ensemble $5 --class_reduced $6

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print
