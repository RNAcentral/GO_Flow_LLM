import polars as pl
import re
import requests
from sklearn.model_selection import train_test_split
from functools import lru_cache
from pathlib import Path
from epmc_xml import fetch

def is_open_access(pmcid):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    paper_url = url.format(pmcid=pmcid)
    r = requests.get(paper_url)
    return r.status_code == 200

@lru_cache
def _get_article(pmcid):
    return fetch.article(pmcid)

def search_protein_id(args):
    pmcid, gene_id = args
    regex = re.compile(f".*{gene_id}.*")
    sect = "discussion"
    article = _get_article(pmcid)
    section_text = article.get_sections()[sect]
    section_paragraphs = section_text.strip().split(" ")
    mentioning_sentences = []
    for para in section_paragraphs:
        if regex.search(para) is not None:
            mentioning_sentences.append(para)
    return mentioning_sentences

@lru_cache
def lookup_rnac_names(rna_id):
    rnacentral_ids = pl.scan_csv("id_mapping.tsv", separator='\t', has_header=False, new_columns=["urs", "source", "external_id", "taxid", "type", "synonym"])
    rnacentral_ids = rnacentral_ids.filter(pl.col("source").is_in(["MIRBASE"]))
    urs, taxid = rna_id.split('_')
    rnc_data = rnacentral_ids.filter((pl.col("urs") == urs) & (pl.col("taxid") == int(taxid))).collect()
    mirbase_id = rnc_data.get_column("external_id").to_list()[0]
    alt_id = rnc_data.get_column("synonym").to_list()[0]
    short_alt = "-".join(alt_id.split("-")[1:3])

    return f"{mirbase_id}|{alt_id}|{short_alt}"


def expand_extension(ext):
    if ext is None:
        return {
        "targets" : list(),
        "anatomical_locations" : list(),
        "cell_lines" : list()
        }
    def get_input(ext_text):
        protein = re.match(r'.*has_input\(UniProtKB:([A-Za-z0-9]+)\)', ext_text)
        if protein:
            protein = protein.group(1)
            return protein
        return None
    
    def get_anatomy(ext_text):
        location = re.match(r'.*occurs_in\(UBERON:([0-9]+)\)', ext_text)
        if location:
            location = location.group(1)
            return f"UBERON:{location}"

        return None

    def get_cell_line(ext_text):
        location = re.match(r'.*occurs_in\(CL:([0-9]+)\)', ext_text)
        if location:
            location = location.group(1)
            return f"CL:{location}"

        return None
    
    proteins = []
    anatomies = []
    cell_lines = []
    for sub_ext in ext.split('|'):
        protein = get_input(sub_ext)
        anatomy = get_anatomy(sub_ext)
        cell_line = get_cell_line(sub_ext)
        proteins.append(protein)
        anatomies.append(anatomy)
        cell_lines.append(cell_line)

    return {
        "targets" : list(set(proteins)),
        "anatomical_locations" : list(set(anatomies)),
        "cell_lines" : list(set(cell_lines))
    }

def assign_classes(df):
    """
    Loop over the dataframe, look at what is known about a paper's annotations and make a classification on that basis
    """
    pmids_done = []
    r_cols = []

    for row in df.iter_rows(named=True):
        if row['pmid'] in pmids_done:
            continue
        
        if row['go_term'] == "GO:0035195":
            rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 1
            else:
                annotation_class = 4
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmids_done.append(row['pmid'])
            rdata["pmid"] = row['pmid']
            r_cols.append(rdata)
        elif row['go_term'] == "GO:0035278":
            rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 3
            else:
                annotation_class = 4
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmids_done.append(row['pmid'])
            rdata["pmid"] = row['pmid']
            r_cols.append(rdata)
        elif row['go_term'] == "GO:0035279":
            rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 2
            else:
                annotation_class = 4
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmids_done.append(row['pmid'])
            rdata["pmid"] = row['pmid']
        
            r_cols.append(rdata)
    return r_cols

    

raw = pl.read_csv("bhf_ucl_annotations.tsv", separator='\t', has_header=False, columns=[1,2,3,4,10], new_columns=["rna_id", "qualifier", "go_term", "pmid", "extension"])
raw = raw.with_columns(pl.col("pmid").str.split(':').list.last())
print(raw)

classification_data = pl.DataFrame(assign_classes(raw))#.filter(pl.col("rna_id") == "URS0000D55DFB_9606"))
## Not sure why this is needed...
classification_data =  classification_data.unique()

pmid_pmcid_mapping = pl.scan_csv(
    "PMID_PMCID_DOI.csv",
)

pmid_go_pmcid = classification_data.lazy().join(
    pmid_pmcid_mapping, left_on="pmid", right_on="PMID"
).filter(pl.col("PMCID").is_not_null())

pmid_go_pmcid = pmid_go_pmcid.with_columns(
    open_access=pl.col("PMCID").map_elements(is_open_access, return_dtype=pl.Boolean)
)


miRNA_articles_oa = pmid_go_pmcid.filter(pl.col("open_access")).collect()
miRNA_articles_oa.write_parquet("miRNA_articles_oa_new.parquet")

print(miRNA_articles_oa)