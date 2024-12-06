#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=5:30:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G   # memory per node
#SBATCH -J "mirna_curator_all_decisions"   # job name
#SBATCH -o "mirna_all_out"   # job output file
#SBATCH -e "mirna_all_err"   # job error file


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source /homes/agreen/.pyenv_setup

module load cuda

pyenv activate mirna-curator
export CUDA_VISIBLE_DEVICES=0
time python src/mirna_curator/test_all_prompts.py \
mirna_curation_prompts.json \
development_set.parquet \
bartowski/Phi-3-medium-128k-instruct-GGUF \
dev_set_with_decisions_1.parquet \
--template phi3-med &

export CUDA_VISIBLE_DEVICES=1
time python src/mirna_curator/test_all_prompts.py \
mirna_curation_prompts.json \
development_set.parquet \
bartowski/Phi-3-medium-128k-instruct-GGUF \
dev_set_with_decisions_2.parquet \
--template phi3-med & 

export CUDA_VISIBLE_DEVICES=2
time python src/mirna_curator/test_all_prompts.py \
mirna_curation_prompts.json \
development_set.parquet \
bartowski/Phi-3-medium-128k-instruct-GGUF \
dev_set_with_decisions_3.parquet \
--template phi3-med &

export CUDA_VISIBLE_DEVICES=3
time python src/mirna_curator/test_all_prompts.py \
mirna_curation_prompts.json \
development_set.parquet \
bartowski/Phi-3-medium-128k-instruct-GGUF \
dev_set_with_decisions_4.parquet \
--template phi3-med 