import polars as pl
import os
import requests
from sklearn.model_selection import train_test_split

def is_open_access(pmcid):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    paper_url = url.format(pmcid=pmcid)
    r = requests.get(paper_url)
    return r.status_code == 200


query = """select 
pmid, 
ontology_term_id as go_term

	from go_term_annotations ann
	join go_term_publication_map pm on ann.go_term_annotation_id = pm.go_term_annotation_id
  join rnc_references ref on ref.id = pm.reference_id
where assigned_by = 'BHF-UCL'
and ontology_term_id in ('GO:0035195', 'GO:0035279', 'GO:0035278')
order by pmid
"""


pmid_go_map = pl.read_database_uri(query, os.getenv("PGDATABASE")).unique()

print(pmid_go_map)

pmid_pmcid_mapping = pl.scan_csv(
    "PMID_PMCID_DOI.csv",
)
print(pmid_pmcid_mapping)

pmid_go_pmcid = pmid_go_map.lazy().join(
    pmid_pmcid_mapping, left_on="pmid", right_on="PMID"
)

pmid_go_pmcid = pmid_go_pmcid.filter(pl.col("PMCID").is_not_null())

pmid_go_pmcid = pmid_go_pmcid.with_columns(
    open_access=pl.col("PMCID").map_elements(is_open_access, return_dtype=pl.Boolean)
)

miRNA_articles_oa = pmid_go_pmcid.filter(pl.col("open_access")).collect()#.write_parquet("miRNA_articles_oa.parquet")

print(miRNA_articles_oa)
dev_set, test_set = train_test_split(miRNA_articles_oa)

dev_set.write_parquet("development_set.parquet")
test_set.write_parquet("test_set.parquet")
## Explode and filter to get relevant GO terms
# all_data = miRNA_articles_oa.explode("go_terms").filter(pl.col("go_term").is_in(['GO:0035195', 'GO:0035279', 'GO:0035278']))


