import polars as pl
import re
import requests
from sklearn.model_selection import train_test_split
from functools import lru_cache
from pathlib import Path
from epmc_xml import fetch
from ratelimit.exception import RateLimitException
import time
def is_open_access(pmcid):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    paper_url = url.format(pmcid=pmcid)
    r = requests.get(paper_url)
    return r.status_code == 200

@lru_cache
def _get_article(pmcid):
    try:
        art = fetch.article(pmcid)
        time.sleep(0.1)
    except RateLimitException:
        print("Ratelimit exceeded, having a 5 second nap")
        time.sleep(5)
        art = fetch.article(pmcid)

    return art

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
    if len(rnc_data) == 0:
        id_string = rna_id
    else:
        mirbase_id = rnc_data.get_column("external_id").to_list()[0]
        alt_id = rnc_data.get_column("synonym").to_list()[0]
        short_alt = "-".join(alt_id.split("-")[1:3])
        id_string = f"{mirbase_id}|{alt_id}|{short_alt}"
    
    return id_string


def identify_used_ids(args):
    print(args)
    pmcid = args['PMCID']
    genes = args['Gene Names']
    rnas = args["rna_id"]
    
    article = _get_article(pmcid)
    full_text = "\n\n".join(list(article.get_sections().values()))
    sentences = full_text.split('.')

    used_rna_id = None
    used_prot_id = None
    if len(rnas) == 1:
        ## Then the URS was unresolved, we should pass 
        ## it back as-is for later manual fixing
        used_rna_id = rnas[0]
    else:
        # Find the most mentioned RNA ID:
        rna_mentions = {rna: 0 for rna in rnas}
        for sentence in sentences:
            for rna in rnas:
                r = re.search(f".*{rna}.*", sentence, re.IGNORECASE)
                if r is not None:
                    rna_mentions[rna] += 1

    if genes[0] is None:
        used_prot_id = "N/A"
    else:
    # Find the most mentioned protein:
        prot_mentions = {prot: 0 for prot in genes}
        for sentence in sentences:
            for prot in genes:
                r = re.search(f".*{prot}.*", sentence, re.IGNORECASE)
                if r is not None:
                    prot_mentions[prot] += 1
                    # print(sentence)
    ## Select the most specific RNA Identifier we can
    ## based on its length

    def select_id(mentions):
        selected_id = None
        for k in sorted(mentions.keys(), key=lambda x: len(x), reverse=True):
            ## Selects the longest key that has nonzero mentions
            if mentions[k] > 0:
                selected_id = k
                break
        ## Returns none if none of the ids was found
        return selected_id

    if used_rna_id is None:
        used_rna_id = select_id(rna_mentions)
    if used_prot_id is None:
        used_prot_id = select_id(prot_mentions)

    return {"used_protein_id": used_prot_id, "used_rna_id": used_rna_id}

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
    pmcids_done = []
    r_cols = []

    for row in df.iter_rows(named=True):
        if row['PMCID'] in pmcids_done:
            continue
        rdata = {}
        rdata["protein_id"] = row["used_protein_id"]
        rdata["rna_id"] = row["used_rna_id"]
        if row['go_term'] == "GO:0035195":
            # rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 1
            else:
                annotation_class = 4
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmcids_done.append(row['PMCID'])
            rdata["PMCID"] = row['PMCID']
            r_cols.append(rdata)
        elif row['go_term'] == "GO:0035278":
            # rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 3
            else:
                annotation_class = 4
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmcids_done.append(row['PMCID'])
            rdata["PMCID"] = row['PMCID']
            r_cols.append(rdata)
        elif row['go_term'] == "GO:0035279":
            # rdata = expand_extension(row["extension"])
            paper_annotations = df.filter(pl.col("pmid") == row['pmid'])
            qualifiers = paper_annotations.get_column("qualifier").to_list()
            go_terms = paper_annotations.get_column("go_term").to_list()
            if 'enables' in qualifiers and 'GO:1903231' in go_terms:
                annotation_class = 2
            else:
                annotation_class = 4 
            rdata["class"] = annotation_class
            rdata['go_term'] = row['go_term']
            pmcids_done.append(row['pmid'])
            rdata["PMCID"] = row['PMCID']
        
            r_cols.append(rdata)
    return r_cols

    

raw = pl.read_csv("bhf_ucl_annotations.tsv", separator='\t', has_header=False, columns=[1,2,3,4,10], new_columns=["rna_id", "qualifier", "go_term", "pmid", "extension"])
raw = raw.with_columns(pl.col("pmid").str.split(':').list.last())
raw = raw.with_columns(res=pl.col("extension").map_elements(expand_extension, return_dtype=pl.Struct)).unnest("res")
# raw = raw.with_columns(rna_names=pl.col("rna_id").map_elements(lookup_rnac_names, return_dtype=pl.String))

targets = raw.unique("pmid").explode("targets").filter(pl.col("targets").is_not_null())

cached_targets = True
if cached_targets and Path("cached_target_data.parquet").exists():
    targets = pl.read_parquet("cached_target_data.parquet")
else:
    uniprot_ids = pl.read_csv("idmapping_uniprot.tsv", separator='\t')
    targets = targets.join(uniprot_ids, left_on="targets", right_on="Entry")
    targets = targets.with_columns(pl.col("Gene Names").str.split(' ')).explode("Gene Names")
    

    pmid_pmcid_mapping = pl.scan_csv(
        "PMID_PMCID_DOI.csv",
    )
    targets = targets.lazy().join(pmid_pmcid_mapping, left_on="pmid", right_on="PMID").filter(pl.col("PMCID").is_not_null()).collect()
    targets = targets.with_columns(
        open_access=pl.col("PMCID").map_elements(is_open_access, return_dtype=pl.Boolean)
    ).filter(pl.col("open_access"))
    targets = targets.with_columns(pl.col("rna_id").map_elements(lookup_rnac_names, return_dtype=pl.String))
    targets.write_parquet("cached_target_data.parquet")

targets = targets.with_columns(pl.col("rna_id").str.split("|")).explode("rna_id")

if not Path("paper_and_targets.csv").exists():
    paper_searching = targets.group_by("PMCID").agg(pl.col("Gene Names").unique(), pl.col("rna_id").unique()).sort(by="PMCID")
    pl.Config.set_tbl_rows(1000)
    print(targets)
    print(paper_searching)

    genes = paper_searching.filter(pl.col("PMCID") == "PMC3735565").get_column("Gene Names").to_list()[0]
    rnas  = paper_searching.filter(pl.col("PMCID") == "PMC3735565").get_column("rna_id").to_list()[0]

    paper_searching = paper_searching.with_columns(res = pl.struct("PMCID", "Gene Names", "rna_id").map_elements(identify_used_ids, return_dtype=pl.Struct))

    paper_searching = paper_searching.unnest("res")
    paper_searching.select(["PMCID", "used_protein_id", "used_rna_id"]).write_csv("paper_and_targets.csv")
else:
    paper_searching = pl.read_csv("paper_and_targets.csv")

print(paper_searching)
print(targets)

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