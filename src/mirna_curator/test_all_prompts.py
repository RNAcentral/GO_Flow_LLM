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
import sqlite3

def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo

def run_one_paper(pmcid, prompts, llm, trace_connection):
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

        llm = prompted_flowchart_step_bool(llm, article.sections[target_section_name], prompt.prompt)

        result_dict[prompt.name] = llm['answer'] == "yes"
    return result_dict



@click.command()
@click.argument("curation_prompts_path")
@click.argument("paper_set_path")
@click.argument("model_name")
@click.argument("output_path")
@click.option("--quant", default="q4_k_m")
@click.option("--template", default="chatml")
@click.option("--trace_storage", default=None)
def main(curation_prompts_path, paper_set_path, model_name, output_path, quant, template, trace_storage):
    curation_prompts_json = open(curation_prompts_path, "r").read()
    prompt_object = CurationPrompts.model_validate_json(curation_prompts_json)

    if trace_storage is not None:
        trace_connection = sqlite3.connect(trace_storage)
        cur = trace_connection.cursor()
        cur.execute("CREATE TABLE go_flow_llm_traces ")
    else:
        trace_connection = None

    # TODO: set this up to use CLI and lookup
    llm = get_model(model_name, chat_template=template, quantization=quant)

    papers = pl.read_parquet(paper_set_path)

    pbar = tqdm(total=len(papers), desc='Running all decisions', colour='green')
    process_one = w_pbar(pbar, partial(run_one_paper, prompts=prompt_object.prompts, llm=llm))

    papers = papers.with_columns(res=pl.col("PMCID").map_elements(process_one, return_dtype=pl.Struct)).unnest("res")

    print(papers)

    papers.write_parquet(output_path)

if __name__ == "__main__":
    main()