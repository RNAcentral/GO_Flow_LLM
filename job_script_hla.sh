#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G   # memory per node
#SBATCH -J "go_flow_llm_updated_fc"   # job name
#SBATCH -o "out_gfllm_updated_fc"   # job output file
#SBATCH -e "err_gfllm_updated_fc"   # job error file
#SBATCH --mail-user=agreen@ebi.ac.uk   # email address
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.pyenv_setup

pyenv activate mirna-curator

python src/mirna_curator/main.py --config configs/curation_config_QwQ_HLA.json

# python src/mirna_curator/main.py --config configs/curation_config_R1-Distill-Llama70B.json

