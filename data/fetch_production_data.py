import polars as pl
import os
from datetime import datetime

query = """
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
HAVING COUNT(DISTINCT lsdb.job_id) > 1
"""

lnc_rna_query = """
SELECT lsr.pmcid as PMCID,
(array_agg(distinct lsdb.job_id)) as rna_ids,
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

prod_data = pl.read_database_uri(lnc_rna_query, os.getenv("PGDATABASE"))
prod_data = prod_data.rename({"pmcid": "PMCID"}).with_row_index()

timestamp = datetime.today().strftime("%Y-%m-%d")

print(prod_data.height)
prod_data.write_parquet(f"lncrna_production_input_data_{timestamp}.parquet")

n_splits = 4
n_per_split = prod_data.height // n_splits
splits = [prod_data.filter(pl.col("index").is_between((i-1)*n_per_split, i*n_per_split)) for i in range(1,n_splits+1)]

for n, s in enumerate(splits):
    s.write_parquet(f"lncrna_production_input_data_{timestamp}_split_{n}.parquet")
