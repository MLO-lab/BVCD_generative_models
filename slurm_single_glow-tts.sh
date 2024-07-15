#!/bin/bash
#SBATCH --job-name=GlowTTS # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=slurm_logs/GlowTTS-%j.out
#SBATCH --error=slurm_logs/GlowTTS-%j.err
##SBATCH -w gpu[25-28]

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

for seed in {0..11}
do
    python glow-tts.py --seed $seed
done

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print