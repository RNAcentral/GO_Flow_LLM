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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

            if config_file is not None and other_params_provided:
                raise click.UsageError(
                    "Config file cannot be used together with other CLI parameters."
                )

            if config_file is None and not other_params_provided:
                raise click.UsageError(
                    "Either config file or CLI parameters must be provided."
                )

            # If config file is provided, read it and update kwargs
            if config_file is not None:
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
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
@click.option("--input_data", help="The input data (PMCID and detected RNA ID) for the process")
@mutually_exclusive_with_config()
def main(config: Optional[str] = None,
         model_path: Optional[str] = None,
         flowchart: Optional[str] = None,
         prompts: Optional[str] = None,
         context_length: Optional[int] = 16384,
         quantization: Optional[str] = None,
         chat_template: Optional[str] = None,
         input_data: Optional[str] = None):
    
    llm = get_model(
        # "bartowski/Phi-3-medium-128k-instruct-GGUF", chat_template="phi3-med", quantization="q4_k_m"
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        chat_template="llama3",
        quantization="q4_k_m",
    )

    article = fetch.article("PMC2760133")

    try:
        cur_flowchart_string = open("mirna_curation_flowchart.json", "r").read()
        cf = curation.CurationFlowchart.model_validate_json(cur_flowchart_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading flowchart, aborting")
        exit()
    try:
        prompt_string = open("mirna_curation_prompts.json", "r").read()
        prompt_data = flow_prompts.CurationPrompts.model_validate_json(prompt_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading prompts, aborting")
        exit()

    graph = ComputationGraph(cf)

    curation_result = graph.execute_graph(llm, article, "let-7c", prompt_data)
    print(curation_result)
    pass


if __name__ == "__main__":
    main()
#
