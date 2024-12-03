"""
Load all the questions and run them all across all the dev set of papers


"""

from mirna_curator.flowchart.flow_prompts import CurationPrompts
from mirna_curator.llm_functions.conditions import prompted_flowchart_step_bool
from mirna_curator.model.llm import get_model
# import click
import polars as pl
from guidance import user, assistant, select, gen

from epmc_xml import fetch

## Fill this out later for testing other models
CHAT_TEMPLATE_LOOKUP = {
    None: None
}

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

# @click.command()
# @click.argument("curation_prompts_path")
# @click.argument("paper_set_path")
# @click.argument("model_name")
def main(curation_prompts_path, paper_set_path, model_name):
    curation_prompts_json = open(curation_prompts_path, "r").read()
    prompt_object = CurationPrompts.model_validate_json(curation_prompts_json)

    # TODO: set this up to use CLI and lookup
    llm = get_model("afg1/phi-3.1-medium", chat_template="phi3-med", quantization="q4_k_m")

    papers = pl.read_parquet(paper_set_path)

    this_paper_results = run_one_paper("PMC5415180", prompt_object.prompts, llm)
    print(this_paper_results)



if __name__ == "__main__":
    main("mirna_curation_prompts.json", "development_set.parquet", "afg1/phi-3.1-medium")