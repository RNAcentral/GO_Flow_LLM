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

prod_data = pl.read_database_uri(query, os.getenv("PGDATABASE"))
prod_data = prod_data.rename({"pmcid": "PMCID"}).with_row_index()

timestamp = datetime.today().strftime("%Y-%m-%d")

print(prod_data.height)
prod_data.write_parquet(f"production_test_data_{timestamp}.parquet")

n_splits = 4
n_per_split = prod_data.height // n_splits
splits = [prod_data.filter(pl.col("index").is_between((i-1)*n_per_split, i*n_per_split)) for i in range(1,n_splits+1)]

for n, s in enumerate(splits):
    s.write_parquet(f"production_test_data_{timestamp}_split_{n}.parquet")
