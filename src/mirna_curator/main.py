# from mirna_curator.apis import litscan
from mirna_curator.model.llm import get_model
from mirna_curator.llm_functions.abstract_filtering import assess_abstract
from mirna_curator.flowchart import curation, flow_prompts
from mirna_curator.flowchart.computation_graph import ComputationGraph
from pydantic import ValidationError
import click
from epmc_xml import fetch
import logging
from functools import wraps
from typing import Optional, Callable
import json
import polars as pl
from mirna_curator.utils.tracing import curation_tracer
from guidance import system, user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import signal
import traceback
import faulthandler
import sys
import time
from pathlib import Path
import os

curation_output = []


def save_handler(signum, frame):
    if curation_output:
        curation_output_df = pl.DataFrame(curation_output)
        curation_output_df.write_parquet("curation_results_partial.parquet")
    if signum == signal.SIGTERM:
        sys.exit(0)


def traceback_handler(signum, frame):
    traceback.print_stack(frame, file=sys.stderr)


signal.signal(signal.SIGUSR1, traceback_handler)
signal.signal(signal.SIGUSR2, save_handler)
signal.signal(signal.SIGTERM, save_handler)


def mutually_exclusive_with_config(config_option: str = "config") -> Callable:
    """
    Decorator to ensure CLI parameters are mutually exclusive with config file.

    Args:
        config_option: Name of the config file option
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            ctx = click.get_current_context()
            config_file = kwargs.get(config_option)

            # Remove config option from kwargs for checking other params
            kwargs_without_config = {
                k: v for k, v in kwargs.items() if k != config_option
            }

            # Check if any other parameters are provided
            other_params_provided = any(
                v is not None for v in kwargs_without_config.values()
            )

            # if config_file is not None and other_params_provided:
            #     raise click.UsageError(
            #         "Config file cannot be used together with other CLI parameters."
            #     )

            if config_file is None and not other_params_provided:
                raise click.UsageError(
                    "Either config file or CLI parameters must be provided."
                )

            # If config file is provided, read it and update kwargs
            if config_file is not None:
                try:
                    with open(config_file, "r") as file_handle:
                        config = json.load(file_handle)
                        # Update kwargs with config values
                        kwargs.update(config)
                except Exception as e:
                    raise click.UsageError(f"Error reading config file: {str(e)}")

            return f(*args, **kwargs)

        return wrapped_f

    return decorator


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to a config.json file with options for the run set",
)
@click.option("--model_path", help="A huggingface ID or local model path")
@click.option("--flowchart", help="The flowchart, defined in JSON")
@click.option("--prompts", help="The prompts, defined in JSON")
@click.option("--context_length", help="The context length for the model")
@click.option("--quantization", help="The quantization for the model")
@click.option("--chat_template", help="The chat template for the model")
@click.option(
    "--input_data", help="The input data (PMCID and detected RNA ID) for the process"
)
@click.option("--output_data", help="The output data (curation result) for the process")
@click.option("--max_papers", help="The maximum number of papers to process")
@click.option(
    "--annot_class", help="Restrict processing to one class of annotation", type=int
)
@click.option(
    "--validate_only",
    help="only load and validate flowcharts",
    is_flag=True,
    default=False,
)
@click.option(
    "--evidence_type",
    help="How to do the evidence extraction",
    type=click.Choice(
        [
            "recursive-paragraph",
            "recursive-sentence",
            "single-sentence",
            "single-paragraph",
            "full-substring",
        ]
    ),
    default="single-sentence",
)
@click.option(
    "--deepseek_mode",
    help="Tweak the reasoning generation for deepseek models",
    is_flag=True,
    default=False,
)
@click.option(
    "--checkpoint_frequency", help="How often to write a results checkpoint", default=-1
)
@click.option(
    "--checkpoint_file_path", help="Name of the file to checkpoint into", default="curation_results_checkpoint.parquet"
)
@click.option(
    "--gpu", help="Which gpu ID to run on, if there are several available", default='0'
)
@click.option(
    "--sampling_parameters_path", help="A JSON file containing the sampling parameters to set", default=None
)
@mutually_exclusive_with_config()
def main(
    config: Optional[str] = None,
    model_path: Optional[str] = None,
    flowchart: Optional[str] = None,
    prompts: Optional[str] = None,
    context_length: Optional[int] = 16384,
    quantization: Optional[str] = None,
    chat_template: Optional[str] = None,
    input_data: Optional[str] = None,
    output_data: Optional[str] = None,
    max_papers: Optional[int] = None,
    annot_class: Optional[int] = None,
    validate_only: Optional[bool] = None,
    evidence_type: Optional[str] = "single-sentence",
    deepseek_mode: Optional[bool] = False,
    checkpoint_frequency: Optional[int] = -1,
    checkpoint_file_path: Optional[str] = None,
    gpu: Optional[str] = None,
    sampling_parameters_path: Optional[str] = None,
):
    curation_tracer.set_model_name(model_path)

    ## Build the run config options dict from things in the config
    run_config_options = {
        "evidence_mode": evidence_type,
        "deepseek_mode": deepseek_mode,
    }
    _flowchart_load_start = time.time()
    try:
        cur_flowchart_string = open(flowchart, "r").read()
        cf = curation.CurationFlowchart.model_validate_json(cur_flowchart_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading flowchart, aborting")
        exit()
    _flowchart_load_end = time.time()
    logger.info(f"Loaded flowchart from {flowchart}")
    logger.info(
        f"Flowchart loaded in {_flowchart_load_end - _flowchart_load_start:.2f} seconds"
    )

    _prompt_load_start = time.time()
    try:
        prompt_string = open(prompts, "r").read()
        prompt_data = flow_prompts.CurationPrompts.model_validate_json(prompt_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading prompts, aborting")
        exit()
    _prompt_load_end = time.time()
    logger.info(f"Loaded prompts from {prompts}")
    logger.info(f"Prompts loaded in {_prompt_load_end - _prompt_load_start:.2f}")

    if validate_only:
        logger.info("Validation only, exiting now")
        return 0
    
    ## Validate arguments - exit if something required is set to None
    if any([
        model_path is None,
        flowchart is None,
        prompts is None,
        input_data is None,
        output_data is None,
        chat_template is None
    ]):
        logger.error("A required argument is se to None, check your config!")
        return 1


    if gpu is not None:
        ## Set which GPU to use
        logger.info("Selecting %s gpu for this process", gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if sampling_parameters_path is not None:
        try:
            sampling_string = open(sampling_parameters_path, 'r').read()
            sampling_parameters = json.loads(sampling_string)
        except Exception as e:
            logger.error(f"failed to fload sampling parameters from {sampling_parameters_path}, with error: {e}")
    else:
        sampling_parameters = {
            "temperature" : 0.6,
            "min_p" : 0.00,
            "top_k" : 40,
            "top_p" : 0.95, # This configuration from danhanchen of Unsloth, should
            "repetition_penalty": 1.1, # reduce the repetition on reasoning
            "dry_multiplier" : 0.5,
        }

    run_config_options.update(sampling_parameters)
    _model_load_start = time.time()
    llm = get_model(
        model_path,
        chat_template=chat_template,
        quantization=quantization,
        context_length=context_length,
        run_config_options=run_config_options,
    )
    _model_load_end = time.time()
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model loaded in {_model_load_end - _model_load_start:.2f} seconds")

    _system_prompt_start = time.time()
    ## Look for a system prompt in the prompts, and apply it if found
    for prompt in prompt_data.prompts:
        if prompt.type == "system":
            logger.info("Found system prompt, applying...")
            try:
                with system():
                    llm += prompt.prompt
            except Exception as e:
                logger.warning(
                    "Selected model does not have a system prompt mode, forward as user instead"
                )
                with user():
                    llm += prompt.prompt

            break
    _system_prompt_end = time.time()
    logger.info(
        f"System prompt (if present) applied in {_system_prompt_end - _system_prompt_start:.2f} seconds"
    )

    _graph_construction_start = time.time()
    graph = ComputationGraph(cf, run_config=run_config_options)
    _graph_construction_end = time.time()
    logger.info("Constructed computation graph")
    logger.info(
        f"Graph constructed in {_graph_construction_end - _graph_construction_start:.2f} seconds"
    )

    ## Get the curation input data and resume if there's a valid checkpoint
    if input_data.endswith("parquet") or input_data.endswith("pq"):
        curation_input = pl.read_parquet(input_data)
    elif input_data.endswith("csv"):
        curation_input = pl.read_csv(input_data)
        ## Split the rna_id column on | to get a list of IDs
        curation_input = curation_input.with_columns(
            pl.col("rna_id").str.split("|").alias("rna_id")
        )
    else:
        logger.error("Unsupported input data format for %s", input_data)
        return 1


    if Path(checkpoint_file_path).exists():
        logger.info("Resuming from checkpoint %s", checkpoint_file_path)
        done = pl.read_parquet(checkpoint_file_path)
        curation_input = curation_input.join(done, on="PMCID", how="anti")
        
    if annot_class is not None:
        logger.info(f"Restricting processing to annotation class {annot_class}")
        curation_input = curation_input.filter(pl.col("class") == annot_class)

    logger.info(f"Loaded input data from {input_data}")
    logger.info(f"Processing up to {curation_input.height} papers")
    logger.info(f"{curation_input.select(pl.col("rna_id").list.len().sum())[0,0]} total graph runs to do (before filtering)")
    ## This is where we start riunning the curation graph for all the papers, one by one.
    _bulk_processing_start = time.time()
    for i, row in enumerate(curation_input.iter_rows(named=True)):
        if max_papers is not None and i >= max_papers:
            break

        ## See if we need to checkpoint, then write output
        if checkpoint_frequency > 0 and i > 0 and i % checkpoint_frequency == 0:
            logger.info("Checkpointing results")
            logger.info(
                f"Curation of {len(curation_output)} articles completed in {time.time()-_bulk_processing_start:.2f} seconds"
            )
            if len(curation_output) > 0:
                curation_output_df = pl.DataFrame(curation_output)
                if Path(checkpoint_file_path).exists():
                    prev = pl.read_parquet(checkpoint_file_path)
                    pl.concat([curation_output_df, prev], how="diagonal_relaxed")
                ## Overwrite the checkpoint to save space
                curation_output_df.write_parquet(checkpoint_file_path)

        try:
            logger.info("Starting curation for paper %s", row["PMCID"])
            _paper_fetch_start = time.time()
            article = fetch.article(row["PMCID"])
            article.add_figures_section()
            _paper_fetch_end = time.time()
        except Exception as e:
            logger.error(e)
            logger.error(f"Failed to fetch/parse {row['PMCID']}, skipping it")
            continue

        logger.info(
            f"Fetched and parsed paper in {_paper_fetch_end - _paper_fetch_start:.2f} seconds"
        )

        _curation_start = time.time()
        if isinstance(row["rna_id"], list):
            rna_ids = row["rna_id"]
        else:
            rna_ids = [row["rna_id"]]

        for rna_id in rna_ids:
            try:
                llm_trace, curation_result = graph.execute_graph(
                    row["PMCID"],
                    llm,
                    article,
                    rna_id,
                    prompt_data,
                )
            except Exception as e:
                logger.error(e)
                logger.error("Paper %s has exceeded context limit, skipping", row["PMCID"])
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                continue
            logger.info(
                f"RNA ID: {row['rna_id']} in {row['PMCID']} - Curation Result: {curation_result}"
            )
            ## Check here if we get filtered, and if so, break the loop and only record filtering
            ## in the curation result. Set RNA id to concatenated string of all ids maybe?
            ## Will require a new terminal node with explicit mention of why no annotation.
            if curation_result["annotation"].get("no_annotation", None) is not None:
                # This means the terminal was a no annotation node, check the reason
                if "filtered" in curation_result["annotation"]["no_annotation"]["reason"]:
                    logger.info(f"RNA {rna_id} filtered, skipping further RNAs in this paper")
                    rna_id = "|".join(rna_ids)
                    curation_result = {
                        "annotation": {
                            "type": "no_annotation",
                            "reason": "filtered",
                        },
                        "evidence": [],
                        "all_reasoning": curation_result.get("all_reasoning", [])
                    }
                    ## Done with this paper
                    curation_output.append(
                    {
                        "PMCID": row["PMCID"],
                        "rna_id": rna_id,
                        "curation_result": curation_result,
                    }
                    )
                    break
            else:
                curation_output.append(
                    {
                        "PMCID": row["PMCID"],
                        "rna_id": rna_id,
                        "curation_result": curation_result,
                    }
                )
        _curation_end = time.time()
        logger.info(
                f"Ran curation graph in {_curation_end - _curation_start:.2f} seconds"
        )
        logger.info(
                f"Curated {len(rna_ids)} RNAs"
        )
        # with open(f"{row['PMCID']}_{row['rna_id']}_llm_trace.txt", "w") as f:
        #     f.write(llm_trace)
    _bulk_processing_end = time.time()
    _bulk_processing_total = _bulk_processing_end - _bulk_processing_start
    _bulk_processing_average = _bulk_processing_total / len(curation_output)
    logger.info(
        f"Curation of {len(curation_output)} articles completed in {_bulk_processing_total:.2f} seconds"
    )
    logger.info(
        f"Average time to curate one paper: {_bulk_processing_average:.2f} seconds"
    )
    curation_output_df = pl.DataFrame(curation_output)
    curation_output_df.write_parquet(output_data)


if __name__ == "__main__":
    main()
#
