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
import sys
import time

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
):
    curation_tracer.set_model_name(model_path)

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

    _model_load_start = time.time()
    llm = get_model(
        model_path,
        chat_template=chat_template,
        quantization=quantization,
        context_length=context_length,
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
    graph = ComputationGraph(cf)
    _graph_construction_end = time.time()
    logger.info("Constructed computation graph")
    logger.info(
        f"Graph constructed in {_graph_construction_end - _graph_construction_start:.2f} seconds"
    )

    curation_input = pl.read_parquet(input_data)
    if annot_class is not None:
        logger.info(f"Restricting processing to annotation class {annot_class}")
        curation_input = curation_input.filter(pl.col("class") == annot_class)

    logger.info(f"Loaded input data from {input_data}")
    logger.info(f"Processing up to {curation_input.height} papers")
    _bulk_processing_start = time.time()
    for i, row in enumerate(curation_input.iter_rows(named=True)):
        if max_papers is not None and i >= max_papers:
            break
        logger.info("Starting curation for paper %s", row["PMCID"])
        _paper_fetch_start = time.time()
        article = fetch.article(row["PMCID"])
        article.add_figures_section()
        _paper_fetch_end = time.time()
        logger.info(
            f"Fetched and parsed paper in {_paper_fetch_end - _paper_fetch_start:.2f} seconds"
        )
        _curation_start = time.time()
        try:
            llm_trace, curation_result = graph.execute_graph(
                row["PMCID"], llm, article, row["rna_id"], prompt_data
            )
        except Exception as e:
            logger.error(e)
            logger.error("Paper %s has exceeded context limit, skipping", row["PMCID"])
            continue
        logger.info(
            f"RNA ID: {row['rna_id']} in {row['PMCID']} - Curation Result: {curation_result}"
        )
        logger.info(
            f"Manual Result - GO term: {row['go_term']}; Protein target: {row['protein_id']}"
        )
        _curation_end = time.time()
        logger.info(
            f"Ran curation graph in {_curation_end - _curation_start:.2f} seconds"
        )
        curation_output.append(
            {
                "PMCID": row["PMCID"],
                "rna_id": row["rna_id"],
                "curation_result": curation_result,
            }
        )
        with open(f"{row['PMCID']}_{row['rna_id']}_llm_trace.txt", "w") as f:
            f.write(llm_trace)
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
