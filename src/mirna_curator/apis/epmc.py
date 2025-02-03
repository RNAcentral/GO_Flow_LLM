import requests
from typing import List

annotations_endpoint_url = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=PMC:{pmcid}&type=Gene_Proteins&provider=Europe PMC"


def get_gene_name_annotations(pmcid: str) -> List[str]:
    """
    Call the EuropePMC API to get gene annotations for a paper. Then
    filter the result to get a sorted, unique list of gene names

    This can then be given to guidance to select from
    """
    res = requests.get(annotations_endpoint_url.format(pmcid=pmcid))

    res.raise_for_status()

    annotations = res.json()[0]["annotations"]

    gene_names = [a["tags"][0]["name"] for a in annotations]

    gene_names = sorted(list(set(gene_names)))
    return gene_names
