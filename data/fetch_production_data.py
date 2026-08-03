import polars as pl
import os
import click
from datetime import datetime
from math import ceil

mirna_query = """
SELECT
    lsr.pmcid AS PMCID,
    (ARRAY_AGG(lsdb.job_id)) AS rna_id,
    COUNT(DISTINCT lsdb.job_id) AS rna_count
FROM litscan_result lsr
JOIN litscan_database lsdb ON lsdb.job_id = lsr.job_id
JOIN litscan_article lsa ON lsa.pmcid = lsr.pmcid
WHERE lsdb.name IN ('mirbase', 'mirgenedb')
    AND lsa.retracted = FALSE
    AND lsa.type = 'Research article'
    AND EXISTS (
        SELECT 1
        FROM litscan_job lsj
        WHERE lsj.job_id = lsdb.job_id
        AND lsj.hit_count > 0
    )
GROUP BY lsr.pmcid
"""

lncrna_query = """
SELECT lsr.pmcid as PMCID,
(array_agg(distinct lsdb.job_id)) as rna_id,
(array_agg(type))[1] as type,
(array_agg(retracted))[1] as retracted

from litscan_result lsr
left join (select distinct lsdb.job_id, lsdb.name from litscan_database lsdb) lsdb
	on lsdb.job_id = lsr.job_id
join litscan_job lsj
	on lsj.job_id = lsdb.job_id
join litscan_article lsa
	on lsa.pmcid = lsr.pmcid
where lsdb.job_id in (
      select distinct job_id  from litscan_database lsdb

    join rnc_accessions ac
      on LOWER(ac.gene) = lsdb.job_id

    where ac.rna_type = 'SO:0001877'
    and lsdb.name = 'genecards'
                     )
and hit_count > 0
and (retracted = false and type = 'Research article')

group by lsr.pmcid"""

QUERIES = {
    "mirna": (mirna_query, "mirna"),
    "lncrna": (lncrna_query, "lncrna"),
}


@click.command()
@click.option(
    "--rna-type",
    type=click.Choice(list(QUERIES)),
    default="mirna",
    help="Which RNA type to pull papers for",
)
@click.option("--n-splits", default=4, help="Number of split files to write")
def main(rna_type, n_splits):
    query, prefix = QUERIES[rna_type]

    prod_data = pl.read_database_uri(query, os.getenv("PGDATABASE"))
    prod_data = prod_data.rename({"pmcid": "PMCID"})

    timestamp = datetime.today().strftime("%Y-%m-%d")
    basename = f"{prefix}_production_input_data_{timestamp}"

    print(prod_data.height)
    prod_data.write_parquet(f"{basename}.parquet")

    ## Round up so the remainder rows land in the last split rather than being dropped
    n_per_split = ceil(prod_data.height / n_splits)
    for n in range(n_splits):
        split = prod_data.slice(n * n_per_split, n_per_split)
        split.write_parquet(f"{basename}_split_{n}.parquet")


if __name__ == "__main__":
    main()
