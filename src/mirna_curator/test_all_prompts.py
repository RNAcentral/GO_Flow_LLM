"""
Load all the questions and run them all across all the dev set of papers


"""

from mirna_curator.flowchart.flow_prompts import CurationPrompts
from mirna_curator.llm_functions.conditions import prompted_flowchart_step_bool
from mirna_curator.model.llm import get_model
import click
import polars as pl
from guidance import user, assistant, select, gen
from functools import partial, wraps
from tqdm import tqdm
from epmc_xml import fetch

def progress_wrapper(func, total=None, desc=None):
    """
    Wraps a function with a tqdm progress bar.
    Works with both regular and partial functions.
    
    Args:
        func: The function to wrap
        total: Total number of iterations (required for partial functions)
        desc: Description for the progress bar
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If first arg is iterable, use its length as total
        if args and hasattr(args[0], '__len__'):
            total_items = len(args[0])
        elif total is not None:
            total_items = total
        else:
            raise ValueError("Must provide either an iterable or specify total")

        # Create progress bar
        pbar = tqdm(total=total_items, desc=desc or func.__name__)
        
        # If first argument is iterable, wrap it with progress bar
        if args and hasattr(args[0], '__len__'):
            new_args = (tqdm(args[0], total=total_items, desc=desc),) + args[1:]
            result = func(*new_args, **kwargs)
        else:
            # For partial functions, update progress manually
            result = func(*args, **kwargs)
            pbar.update(total_items)
        
        pbar.close()
        return result
    
    return wrapper

def run_one_paper(pmcid, prompts, llm):
    article = fetch.article(pmcid)
    result_dict = {}
    for prompt in prompts:
        if prompt.type.startswith("terminal"): ## for now
            continue

        if not prompt.target_section in article.sections.keys():
            check_subtitles = [
                prompt.target_section in section_name
                for section_name in article.sections.keys()
            ]
            if not any(check_subtitles):
                with user():
                    llm += (
                        f"We are looking for the closest section heading to {prompt.target_section} from "
                        f"the following possbilities: {article.sections.keys()}. Which one is closest?"
                    )
                with assistant():
                    llm += select(
                        article.sections.keys(), name="target_section_name"
                    )
                target_section_name = llm["target_section_name"]
            else:
                target_section_name = list(article.sections.keys())[
                    check_subtitles.index(True)
                ]
        else:
            target_section_name = prompt.target_section

        result = prompted_flowchart_step_bool(llm, article.sections[target_section_name], prompt.prompt)

        result_dict[prompt.name] = result
    return result_dict



@click.command()
@click.argument("curation_prompts_path")
@click.argument("paper_set_path")
@click.argument("model_name")
@click.argument("output_path")
@click.option("--quant", default="q4_k_m")
@click.option("--template", default="chatml")
def main(curation_prompts_path, paper_set_path, model_name, output_path, quant, template):
    curation_prompts_json = open(curation_prompts_path, "r").read()
    prompt_object = CurationPrompts.model_validate_json(curation_prompts_json)

    # TODO: set this up to use CLI and lookup
    llm = get_model(model_name, chat_template=template, quantization=quant)

    papers = pl.read_parquet(paper_set_path)

    process_one = progress_wrapper(partial(run_one_paper, prompts=prompt_object.prompts, llm=llm), total=papers.height, desc="Running all decisions...")

    papers = papers.with_columns(res=pl.col("PMCID").map_elements(process_one)).unnest("res")

    print(papers)

    papers.write_parquet(output_path)

if __name__ == "__main__":
    main()