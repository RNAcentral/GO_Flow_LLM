#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=7:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G   # memory per node
#SBATCH -J "go_flow_llm"   # job name
#SBATCH -o "out_gfllm"   # job output file
#SBATCH -e "err_gfllm"   # job error file
#SBATCH --mail-user=agreen@ebi.ac.uk   # email address
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.pyenv_setup

pyenv activate mirna-curator

python src/mirna_curator/main.py --config curation_config_QwQ.json

