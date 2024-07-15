#!/bin/bash
#SBATCH --time=54:00:00 # maximum allocated time
#SBATCH --job-name=wav # name of the job
#SBATCH -p gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=slurm_logs/wav-%j.out
#SBATCH --error=slurm_logs/wav-%j.err
##SBATCH --nodes=4
##SBATCH --tasks-per-node=4
##SBATCH -w gpu[25,27,28,30]
##SBATCH -N 5
##SBATCH --mem=3000
##SBATCH -a 0-120%3

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python glow-tts_generate.py --epoch $1 --seed $2
	
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print
