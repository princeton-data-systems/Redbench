#!/usr/bin/env python3
"""Screen provisioned Redset clusters for usable SELECTs.

Identifies clusters with high volumes of SELECTs that have non-null
read_table_ids and mbytes_scanned -- the requirements for Redbench generation.
"""
import duckdb
import sys

REDSET = "work/data/full_provisioned.parquet"

con = duckdb.connect()

print("Loading provisioned dataset (this may take a minute)...")
print(f"File: {REDSET}")

# First: overall stats
result = con.execute(f"""
    SELECT 
        COUNT(*) as total_queries,
        COUNT(DISTINCT instance_id) as num_clusters,
        COUNT(DISTINCT (instance_id, database_id)) as num_cluster_db_pairs
    FROM read_parquet('{REDSET}')
""").fetchone()
print(f"\nOverall: {result[0]:,} queries, {result[1]} clusters, {result[2]} cluster/db pairs")

# Query type breakdown
print("\nQuery type breakdown:")
rows = con.execute(f"""
    SELECT query_type, COUNT(*) as cnt
    FROM read_parquet('{REDSET}')
    GROUP BY query_type
    ORDER BY cnt DESC
""").fetchall()
for qt, cnt in rows:
    print(f"  {qt}: {cnt:,}")

# Per-cluster stats: total queries, total SELECTs, usable SELECTs
print("\n\nTop clusters by usable SELECTs (non-null read_table_ids AND mbytes_scanned):")
print(f"{'Cluster':>8} {'DB':>4} {'Total':>12} {'SELECTs':>10} {'Usable':>10} {'Usable%':>8}")
print("-" * 60)

rows = con.execute(f"""
    SELECT 
        instance_id,
        database_id,
        COUNT(*) as total,
        SUM(CASE WHEN query_type = 'select' THEN 1 ELSE 0 END) as selects,
        SUM(CASE WHEN query_type = 'select' 
                  AND read_table_ids IS NOT NULL 
                  AND mbytes_scanned IS NOT NULL 
             THEN 1 ELSE 0 END) as usable
    FROM read_parquet('{REDSET}')
    GROUP BY instance_id, database_id
    HAVING usable > 0
    ORDER BY usable DESC
    LIMIT 40
""").fetchall()

for inst, db, total, selects, usable in rows:
    pct = 100 * usable / selects if selects > 0 else 0
    print(f"{inst:>8} {db:>4} {total:>12,} {selects:>10,} {usable:>10,} {pct:>7.1f}%")

# Also show the time range for top clusters
print("\n\nTime range for top 10 clusters:")
top_clusters = [(r[0], r[1]) for r in rows[:10]]
for inst, db in top_clusters:
    tr = con.execute(f"""
        SELECT 
            MIN(arrival_timestamp) as min_ts,
            MAX(arrival_timestamp) as max_ts,
            DATEDIFF('day', MIN(arrival_timestamp), MAX(arrival_timestamp)) as days
        FROM read_parquet('{REDSET}')
        WHERE instance_id = {inst} AND database_id = {db}
    """).fetchone()
    print(f"  Cluster {inst}/{db}: {tr[0]} to {tr[1]} ({tr[2]} days)")

con.close()
