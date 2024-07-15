#!/bin/bash
#SBATCH --time=10:00:00 # maximum allocated time
#SBATCH --job-name=wav-allJobs # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=slurm_logs/wav-allJobs-%j.out
#SBATCH --error=slurm_logs/wav-allJobs-%j.err

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

#for seed in $(seq 0 1 19)
for epoch in $(seq 24000 2000 36000)
do
    for seed in 2 5 8 11
    do
        sbatch wav_generate.sh $epoch $seed
        sleep 10
	done
done

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print