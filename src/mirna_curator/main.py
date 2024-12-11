# from mirna_curator.apis import litscan
from mirna_curator.model.llm import get_model
from mirna_curator.llm_functions.abstract_filtering import assess_abstract
from mirna_curator.flowchart import curation, flow_prompts
from mirna_curator.flowchart.computation_graph import ComputationGraph

from epmc_xml import fetch


def main():
    # pmcids, abstracts = litscan._fetch_all_results(
    #     "hsa-miR-5189-5p", include_abstracts=True
    # )

    llm = get_model(
        "bartowski/Phi-3-medium-128k-instruct-GGUF", chat_template="phi3-med", quantization="q4_k_m"
    )

    # abstract = abstracts[0]

    # abstract = "MicroRNAs are a class of small non-coding RNAs and participate in the regulation of apoptotic program. Although miR-21 is able to inhibit apoptosis, its expression regulation and downstream targets remain to be fully elucidated. Here we report that the transcriptional factor Foxo3a initiates apoptosis by transcriptionally repressing miR-21 expression. Our results showed that doxorubicin could simultaneously induce the translocation of Foxo3a to the cell nuclei and a reduction in miR-21 expression. Knockdown of Foxo3a resulted in an elevation in miR-21 levels, whereas enforced expression of Foxo3a led to a decrease in miR-21 expression. In exploring the molecular mechanism by which Foxo3a regulates miR-21, we observed that Foxo3a bound to the promoter region of miR-21 and suppressed its promoter activity. These results indicate that Foxo3a can transcriptionally repress miR-21 expression. In searching for the downstream targets of miR-21 in apoptosis, we found that miR-21 suppressed the translation of Fas ligand (FasL), a pro-apoptotic factor. Furthermore, Foxo3a was able to up-regulate FasL expression through down-regulating miR-21. Our data suggest that Foxo3a negatively regulates miR-21 in initiating apoptosis."

    # print(assess_abstract(llm, abstract))

    # print(pmcids[0])

    article = fetch.article("PMC4113387")


    cf = curation.CurationFlowchart.parse_file("mirna_curation_flowchart.json")
    prompts = flow_prompts.CurationPrompts.parse_file("mirna_curation_prompts.json")

    graph = ComputationGraph(cf)

    graph.execute_graph(llm, article, "miR-10a", prompts)
    pass


if __name__ == "__main__":
    main()
