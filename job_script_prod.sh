#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G   # memory per node
#SBATCH -J "go_flow_llm_production_run"   # job name
#SBATCH -o "out_gfllm_prod"   # job output file
#SBATCH -e "err_gfllm_prod"   # job error file
#SBATCH --mail-user=agreen@ebi.ac.uk   # email address
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.pyenv_setup
export HF_HUB_ENABLE_HF_TRANSFER=1
pyenv activate mirna-curator

# python src/mirna_curator/main.py --config configs/curation_config_QwQ_HLA.json

python src/mirna_curator/main.py --config configs/curation_config_QwQ_prod.json --checkpoint_file_path gfllm_qwq_checkpoint_split_0.parquet --input_data production_test_data_2025-03-31_split_0.parquet --output_data production_data_output_2025-03-31_chunk_0.parquet --gpu 0 &
python src/mirna_curator/main.py --config configs/curation_config_QwQ_prod.json --checkpoint_file_path gfllm_qwq_checkpoint_split_1.parquet --input_data production_test_data_2025-03-31_split_1.parquet --output_data production_data_output_2025-03-31_chunk_1.parquet --gpu 1 &
python src/mirna_curator/main.py --config configs/curation_config_QwQ_prod.json --checkpoint_file_path gfllm_qwq_checkpoint_split_2.parquet --input_data production_test_data_2025-03-31_split_2.parquet --output_data production_data_output_2025-03-31_chunk_2.parquet --gpu 2 &
python src/mirna_curator/main.py --config configs/curation_config_QwQ_prod.json --checkpoint_file_path gfllm_qwq_checkpoint_split_3.parquet --input_data production_test_data_2025-03-31_split_3.parquet --output_data production_data_output_2025-03-31_chunk_3.parquet --gpu 3 

wait