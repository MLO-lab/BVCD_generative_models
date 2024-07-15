#!/bin/bash
#SBATCH --job-name=MNIST-DDPM # name of the job
#SBATCH -p gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./slurm_logs/MNIST-DDPM-%j.out
#SBATCH --error=./slurm_logs/MNIST-DDPM-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=5
##SBATCH -w gpu[27,28]
##SBATCH --mem-per-gpu=10000
##SBATCH --mem=3000

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

## generate samples of single diffusion model
python3 generate_samples.py --epoch $1 --model_folder $2 --save_folder $3 --n_sample $4 --n_models $5
	
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print
