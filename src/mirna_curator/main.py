# from mirna_curator.apis import litscan
from mirna_curator.model.llm import get_model
from mirna_curator.llm_functions.abstract_filtering import assess_abstract
from mirna_curator.flowchart import curation, flow_prompts
from mirna_curator.flowchart.computation_graph import ComputationGraph
from pydantic import ValidationError

from epmc_xml import fetch


def main():
    llm = get_model(
        # "bartowski/Phi-3-medium-128k-instruct-GGUF", chat_template="phi3-med", quantization="q4_k_m"
        "bartowski/Llama-3.3-70B-Instruct-GGUF", chat_template="llama3", quantization="q4_k_m"
    )

    article = fetch.article("PMC2760133")

    try:
        cur_flowchart_string = open("mirna_curation_flowchart.json", 'r').read()
        cf = curation.CurationFlowchart.model_validate_json(cur_flowchart_string)
    except ValidationError as e:
        print(e)
        exit()
    try:
        prompt_string = open("mirna_curation_prompts.json", 'r').read()
        prompts = flow_prompts.CurationPrompts.model_validate_json(prompt_string)
    except ValidationError as e:
        print(e)
        exit()


    graph = ComputationGraph(cf)

    curation_result = graph.execute_graph(llm, article, "let-7c", prompts)
    print(curation_result)
    pass


if __name__ == "__main__":
    main()
# 