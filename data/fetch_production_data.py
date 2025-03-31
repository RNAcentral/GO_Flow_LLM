import polars as pl
import os
from datetime import datetime

query = """
select

pmcid as PMCID,
(array_agg(lsdb.job_id))[1] as rna_id

from litscan_result lsr
left join (select distinct lsdb.job_id, lsdb.name from litscan_database lsdb) lsdb
	on lsdb.job_id = lsr.job_id
join litscan_job lsj 
	on lsj.job_id = lsdb.job_id
where name in ('mirbase', 'mirgenedb')
and hit_count > 0

group by pmcid
having cardinality(array_agg(lsdb.job_id)) = 1
"""

prod_data = pl.read_database_uri(query, os.getenv("PGDATABASE"))
prod_data = prod_data.rename({"pmcid": "PMCID"})

timestamp = datetime.today().strftime("%Y-%m-%d")

prod_data.write_parquet(f"production_test_data_{timestamp}.parquet")
