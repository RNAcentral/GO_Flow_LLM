"""
This is not a unit test, but instead should be a validation of prompting/LLM orchestration test

Here, I'm aiming to assess the False Negative rate my prompts produce when I run them cross a set of abstracts
we _know_ should be included.

There are 50 abstracts extracted from the PDFs of 50 papers that were curated by the UCL-BHF team. All of them should be selected
"""

import polars as pl

from mirna_curator.model.llm import get_model
from mirna_curator.llm_functions.abstract_filtering import assess_abstract


llm = get_model("afg1/phi-3.1-medium", chat_template="phi3-med", quantization="q4_k_m")
ucl_abstracts = pl.read_parquet("UCL_BHF_abstracts.parquet")


ucl_abstracts_with_classification = ucl_abstracts.with_columns(
    llm_decision=pl.col("abstract").map_elements(
        lambda x: assess_abstract(llm, x), return_dtype=pl.Boolean
    )
)

ucl_abstracts_with_classification.write_parquet("classified_abstracts_phi3-med.parquet")

print(ucl_abstracts_with_classification.group_by("llm_decision").agg(pl.len()))
