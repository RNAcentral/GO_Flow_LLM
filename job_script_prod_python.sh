#!/bin/bash

#Submit this script with: sbatch job_script_prod_python.sh
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G   # memory per node
#SBATCH -J "go_flow_llm_production_run_python"   # job name
#SBATCH -o "out_gfllm_prod_python"   # job output file
#SBATCH -e "err_gfllm_prod_python"   # job error file
#SBATCH --mail-user=agreen@ebi.ac.uk   # email address
#SBATCH --mail-type=END

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.pyenv_setup
export HF_HUB_ENABLE_HF_TRANSFER=1
pyenv activate mirna-curator

# Run the Python controller script instead of individual processes
python parallel_controller.py \
    --config configs/curation_config_QwQ_prod.json \
    --gpu-count 4 \
    --checkpoint-pattern "gfllm_qwq_checkpoint_split_{}.parquet" \
    --input-pattern "production_test_data_2025-03-31_split_{}.parquet" \
    --output-pattern "production_data_output_2025-03-31_chunk_{}.parquet" \
    --check-interval 60 \
    --log-dir "logs"
