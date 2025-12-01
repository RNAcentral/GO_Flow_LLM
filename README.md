## GoFlowLLM

An LLM driven agent to follow the miRNA cuation flowchart developed by the UCL curation group. This agent uses reasoning-enabled LLMs to answer relatively simple yes/no questions about a paper that guide it to curate in a highly consistent way suitable for inclusion in the Gene Ontology knowledgebase.

GoFlowLLM is designed first to automate the curation of miRNA mediated gene silencing, but qorks with a flowchart system that should be generic enough to allow other types of curation as well. 


## Prerequisites

This tool will work locally, if you have a powerful enough machine. For our testing, we used the Qwen QwQ32B model at 8-bit quantization. This is _just_ about runnable on a 32GB M1 macbook pro, but not really. For local development/testing I would suggest using a lower quantization (4-bit is probably ok) or a smaller model (e.g. one of the 14B distillations of DeepSeekR1).

If you want to run for real, a single A100-80GB will comfortably hold the QwQ32B model with ~164k token context which should be more than enough for this curation task. 

Note that GoFLowLLM is designed from the ground up to be local-first. This means there is no way for you to simply plug an API key in and use OpenAI, Anthropic or Google's models. There are good reasons for this, which you can read about in the paper! 

## Setting Up

When developiong the tool, we used `pyenv` and `pyenv-virtualenv` for python virtual environment management. You can set up an environment with 
```
pyenv install 3.12.2
pyenv virtualenv 3.12.2 go-flow-llm
```

This will create a pyenv managed virtualenv for you to run the project in. Activate it with `pyenv activate go-flow-llm`.

To install the required dependencies and the GoFlowLLM code:

```
git clone https://github.com/RNAcentral/GO_Flow_LLM.git
cd GO_Flow_LLM
pyenv activate go-flow-llm # If you didn't already
pip install -e .
```

this will install the package (which is still called mirna-curator) into the virtual environment along with all required dependencies. You will also be able to edit the GoFlowLLM code and have it run if you choose to do any development.

## Making a run

There are a few things you will need to prepare to be able to run GoFlowLLM:

1. A flowchart/prompt pair. These are defined in JSON format. See the examples at https://github.com/RNAcentral/GO_Flow_LLM/tree/main/flowcharts/GO_flowchart_2025_production for the current GO miRNA curation flowchart in GOFlow format. Alternatively, there is a prototype flowchart building tool available here: https://afg1.github.io/llm_flow_builder/ This is very much a work in progress, but hopefully will help get a simple flowchart built quickly. See [docs/flowcharts.md](docs/flowcharts.md) documentation for full details
2. A Config file. This is another JSON file defining how we will run the model. See https://github.com/RNAcentral/GO_Flow_LLM/blob/main/configs/curation_config_QwQ_HLA.json for the configuration used to run the curation in the paper. The key parts of this object are the model path (which in this case is a huggingface model ID), the paths for flowchart and prompts, and the input/output data paths. See the documentation in [docs/configuration.md](docs/configuration.md) for details on writing a configuration file and full details of the options available.
3. Input data. This is relatively simple to prepare. It should be a two-column table with a column called `PMCID` containing open-access articles that you want to curate and a column `rna_id` containing the human-readable RNA ID you expect in the paper (e.g. hsa-mir-121). The formats accepted are csv or parquet.

Once you have all these, you should be able to make a run. You will need a job script to orchestrate things, here is an example of one we used during testing:
```
#!/bin/bash
#SBATCH --time=9:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G   # memory per node
#SBATCH -J "go_flow_llm_production_run"   # job name
#SBATCH -o "out_gfllm_prod"   # job output file
#SBATCH -e "err_gfllm_prod"   # job error file
#SBATCH --mail-user=   # email address
#SBATCH --mail-type=END

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.pyenv_setup
export HF_HUB_ENABLE_HF_TRANSFER=1
pyenv activate go-flow-llm

python src/mirna_curator/main.py --config configs/curation_config_QwQ_HLA.json
```

Clearly, this is a slurm job submission script, since that is what we use in our HPC environment. In this, we request a single A100 GPU and 9 hours of runtime. All of the critical configuration is done in the config JSON file.

### Doing bigger runs

One GPU limits the throughput of the system, and this is an embarassingly parallel problem (curation result on one paper _shouldn't_ affect curation on another), so it is trivial to parallelise. We provide a utility script `parallel_controller.py` to manage this, which allows for running on multiple GPUs concurrently with python multiprocessing. Here is the job script we used to curate 6,996 papers in 58 hours:

```
#!/bin/bash
#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=24   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G   # memory per node
#SBATCH -J "go_flow_llm_production_run_python"   # job name
#SBATCH -o "out_gfllm_prod_python"   # job output file
#SBATCH -e "err_gfllm_prod_python"   # job error file
#SBATCH --mail-user=   # email address
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
```

This is very similar to the one-process job, but requests 4 GPUs, and launches the jobs using the parallel controller. Note that we pre-split the input dataset into 4 equal chunks, and will recombine the 4 output files after the run is completed. The parallel controller also independantly checkpoints results from each process, allowing us to resume when something fails.