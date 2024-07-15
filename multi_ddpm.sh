#!/bin/bash
#SBATCH --job-name=MNIST-DDPM-allJobs # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./slurm_logs/MNIST-DDPM-allJobs-%j.out
#SBATCH --error=./slurm_logs/MNIST-DDPM-allJobs-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

# # train diffusion models
# n_epoch=51
# frac=1.0
# deep_ensemble=false
# model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
# save_folder="models/infimnist/ddpm/generated_frac${frac}/"
# for run_id in $(seq 0 31)
# do
#     sbatch slurm_train_ddpm.sh $run_id $n_epoch $frac $model_folder $deep_ensemble
#     sleep 10
# done

# # generate samples for trained models
# for n_epoch in $(seq 0 2 50)
# do
#     sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder
#     sleep 10
# done

# # train diffusion models with number of zeros reduced to X%
# n_epoch=51
# deep_ensemble=false
# for frac in $(seq 0.2 0.2 0.8)
# do
#     model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
#     save_folder="models/infimnist/ddpm/generated_frac${frac}/"
#     for run_id in $(seq 0 19)
#     do
#         sbatch slurm_train_ddpm.sh $run_id $n_epoch $frac $model_folder $deep_ensemble
#         sleep 10
#     done
# done

# # generate samples for trained models
# for frac in $(seq 0.2 0.2 0.8)
# do
#     model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
#     save_folder="models/infimnist/ddpm/generated_frac${frac}/"
#     for n_epoch in $(seq 0 2 50)
#     do
#         sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder
#         sleep 10
#     done
# done

# # train diffusion models with number of class i reduced to 1%
# n_epoch=41
# deep_ensemble=false
# frac=0.01
# for class_reduced in $(seq 2 3)
# do
#     model_folder="models/infimnist/ddpm/setid{}_frac${frac}_class${class_reduced}/"
#     save_folder="models/infimnist/ddpm/generated_frac${frac}_class${class_reduced}/"
#     for run_id in $(seq 0 19)
#     do
#         sbatch slurm_train_ddpm.sh $run_id $n_epoch $frac $model_folder $deep_ensemble $class_reduced
#         sleep 10
#     done
# done

# generate samples for trained models
frac=0.01
# 10 per class
n_sample=100
n_models=10
for class_reduced in $(seq 2 3)
do
    model_folder="models/infimnist/ddpm/setid{}_frac${frac}_class${class_reduced}/"
    save_folder="models/infimnist/ddpm/generated_frac${frac}_class${class_reduced}/"
    for n_epoch in $(seq 0 2 40)
    do
        sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder $n_sample $n_models
        sleep 10
    done
done

# # ablation study: deep ensembles (same train set, different weight inits)
# n_epoch=51
# frac=1.0
# model_folder="models/infimnist/ddpm/setid0_modelid{}_frac${frac}/"
# save_folder="models/infimnist/ddpm/generated_frac${frac}_DE/"
# deep_ensemble=true
# for run_id in $(seq 0 19)
# do
#     sbatch slurm_train_ddpm.sh $run_id $n_epoch $frac $model_folder $deep_ensemble
#     sleep 10
# done

# # generate samples for trained models
# for n_epoch in $(seq 0 2 50)
# do
#     sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder
#     sleep 10
# done

# train diffusion models with number of zeros reduced to X%
#n_epoch=51
#deep_ensemble=false
#for frac in $(seq 0.01 0.01 0.05)
# for frac in 0.1
# do
#     model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
#     save_folder="models/infimnist/ddpm/generated_frac${frac}/"
#     for run_id in $(seq 0 19)
#     do
#         sbatch slurm_train_ddpm.sh $run_id $n_epoch $frac $model_folder $deep_ensemble
#         sleep 10
#     done
# done

# generate samples for trained models
# for frac in $(seq 0.01 0.01 0.05)
# do
#     model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
#     save_folder="models/infimnist/ddpm/generated_frac${frac}/"
#     for n_epoch in $(seq 0 2 50)
#     do
#         sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder
#         sleep 10
#     done
# done

# frac=0.1
# model_folder="models/infimnist/ddpm/setid{}_frac${frac}/"
# save_folder="models/infimnist/ddpm/generated_frac${frac}/"
# for n_epoch in $(seq 0 2 50)
# do
#     sbatch slurm_generate_samples.sh $n_epoch $model_folder $save_folder
#     sleep 10
# done


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print
