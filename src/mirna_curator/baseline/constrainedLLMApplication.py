"""
This is a baseline to compare the flowchart against. To make it a slightly fairer comparison, we will
use constrained decoding as well, but I may run one to see what GO term the LLM comes up with if I don't force it.

The plan then:

0. Load the LLM with min. 32k context
1. Load the list of papers
 foreach paper:
    - load whole paper text
    - Expand prompt template
    - Reason for 1024 tokens
    - Select answer - constrained to 3 we want
"""
import polars as pl
import click
from mirna_curator.model.llm import get_model
from functools import wraps
from epmc_xml import fetch
import json
from typing import Optional, Callable
import time
import logging
from guidance import user, assistant, gen, select
from mirna_curator.model.llm import STOP_TOKENS
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_template = """You are curating a paper for the miRNA-mRNA interaction it contains.
To do this, you will evaluate the presented evidence, then choose from one of the following 
GO terms:

GO:0035195 - miRNA-mediated post-transcriptional gene silencing - A post-transcriptional gene silencing pathway in which regulatory microRNAs (miRNAs) elicit silencing of specific target genes. miRNAs are endogenous 21-24 nucleotide small RNAs processed from stem-loop RNA precursors (pre-miRNAs). Once incorporated into a RNA-induced silencing complex (RISC), miRNAs can downregulate protein production by either of two posttranscriptional mechanisms: endonucleolytic cleavage of the RNA (often mRNA) or mRNA translational repression, usually accompanied by poly-A tail shortening and subsequent degradation of the mRNA. miRNAs are present in all the animals and in plants, whereas siRNAs are present in lower animals and in plants. PMID:14744438 PMID:15066275 PMID:15066283 PMID:23209154 PMID:23985560 PMID:28379604 

GO:0035278 - miRNA-mediated gene silencing by inhibition of translation - An RNA interference pathway in which microRNAs (miRNAs) block the translation of target mRNAs into proteins. Once incorporated into a RNA-induced silencing complex (RISC), a miRNA will typically mediate repression of translation if the miRNA imperfectly base-pairs with the 3' untranslated regions of target mRNAs. PMID:14744438 PMID:15196554 

GO:0035279 - miRNA-mediated gene silencing by mRNA destabilization - An RNA interference pathway in which microRNAs (miRNAs) direct the cleavage of target mRNAs. Once incorporated into a RNA-induced silencing complex (RISC), a miRNA base pairing with near-perfect complementarity to the target mRNA will typically direct targeted endonucleolytic cleavage of the mRNA. Many plant miRNAs downregulate gene expression through this mechanism. PMID:14744438 PMID:15196554 PMID:21118121 PMID:23209154 

You must carefully consider the experimental evidence given, and what must be shown to give an annotation to each of these terms.

Here is the fulltext of the paper:
--------
{paper_text}
--------

Having read the paper, recall the three GO terms we are annotating to:
GO:0035195 - miRNA-mediated post-transcriptional gene silencing - A post-transcriptional gene silencing pathway in which regulatory microRNAs (miRNAs) elicit silencing of specific target genes. miRNAs are endogenous 21-24 nucleotide small RNAs processed from stem-loop RNA precursors (pre-miRNAs). Once incorporated into a RNA-induced silencing complex (RISC), miRNAs can downregulate protein production by either of two posttranscriptional mechanisms: endonucleolytic cleavage of the RNA (often mRNA) or mRNA translational repression, usually accompanied by poly-A tail shortening and subsequent degradation of the mRNA. miRNAs are present in all the animals and in plants, whereas siRNAs are present in lower animals and in plants. PMID:14744438 PMID:15066275 PMID:15066283 PMID:23209154 PMID:23985560 PMID:28379604 

GO:0035278 - miRNA-mediated gene silencing by inhibition of translation - An RNA interference pathway in which microRNAs (miRNAs) block the translation of target mRNAs into proteins. Once incorporated into a RNA-induced silencing complex (RISC), a miRNA will typically mediate repression of translation if the miRNA imperfectly base-pairs with the 3' untranslated regions of target mRNAs. PMID:14744438 PMID:15196554 

GO:0035279 - miRNA-mediated gene silencing by mRNA destabilization - An RNA interference pathway in which microRNAs (miRNAs) direct the cleavage of target mRNAs. Once incorporated into a RNA-induced silencing complex (RISC), a miRNA base pairing with near-perfect complementarity to the target mRNA will typically direct targeted endonucleolytic cleavage of the mRNA. Many plant miRNAs downregulate gene expression through this mechanism. PMID:14744438 PMID:15196554 PMID:21118121 PMID:23209154 

Now think about what evidence is needed to annotate to each term, then select your annotation. Alternatively, if no evidence is given you may choose to select 'No Annotation'
"""

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


def do_curation_constrained(llm, completed_prompt, choices):
    with user():
        llm += completed_prompt

    with assistant():
        llm += "<think>" + gen('reasoning', max_tokens=1024, stop=STOP_TOKENS)

        llm += "\nTherefore the most appropriate choice is " + select(choices, name="annotation")

    return llm['annotation'], llm['reasoning']

@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to a config.json file with options for the run set",
)
@click.option("--model_path", help="A huggingface ID or local model path")
@click.option("--context_length", help="The context length for the model")
@click.option("--quantization", help="The quantization for the model")
@click.option("--chat_template", help="The chat template for the model")
@click.option(
    "--input_data", help="The input data (PMCID and detected RNA ID) for the process"
)
@click.option("--output_data", help="The output data (curation result) for the process")
@click.option(
    "--deepseek_mode",
    help="Tweak the reasoning generation for deepseek models",
    is_flag=True,
    default=False,
)
@mutually_exclusive_with_config()
def main(config: Optional[str] = None,
        model_path: Optional[str] = None,
        context_length: Optional[int] = 16384,
        quantization: Optional[str] = None,
        chat_template: Optional[str] = None,
        deepseek_mode: Optional[bool] = False,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None,
        ):

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



    curation_input = pl.read_parquet(input_data)
    if Path(output_data).exists():
        logger.info("Resuming from checkpoint %s", output_data)
        done = pl.read_parquet(output_data)
        curation_input = curation_input.join(done, on="PMCID", how="anti")

    logger.info(f"Loaded input data from {input_data}")
    logger.info(f"Processing up to {curation_input.height} papers")

    go_terms = ["GO:0035195", "GO:0035278", "GO:0035279", "No Annotation"]
    curation_output = []

    _bulk_processing_start = time.time()
    for i, row in enumerate(curation_input.iter_rows(named=True)):
        try:
            logger.info("Starting curation for paper %s", row["PMCID"])
            _paper_fetch_start = time.time()
            article = fetch.article(row["PMCID"])
            article.add_figures_section()
            _paper_fetch_end = time.time()
        except:
            logger.error(f"Failed to fetch/parse {row['PMCID']}, skipping it")
            continue
        
        logger.info(
            f"Fetched and parsed paper in {_paper_fetch_end - _paper_fetch_start:.2f} seconds"
        )

        _curation_start = time.time()
        completed_prompt = prompt_template.format(paper_text=article.get_body())

        annotation, reasoning = do_curation_constrained(llm, completed_prompt, go_terms)
        
        _curation_end = time.time()
        logger.info(
            f"Selected term in {_curation_end - _curation_start:.2f} seconds"
        )
        curation_output.append(
            {
                "PMCID": row["PMCID"],
                "rna_id": row["rna_id"],
                "annotation": annotation,
                "reasoning": reasoning,
            }
        )
        logger.info(curation_output[-1])
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