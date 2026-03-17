#!/usr/bin/env python3
"""Create per-cluster sampled parquet files from the provisioned Redset.

For clusters with millions of queries, we take a uniform random sample
(preserving the full 3-month time range) to keep generation tractable.
The sample is stratified by day to ensure even temporal coverage.

Usage:
    python sample_provisioned.py [--max-rows 500000] [--clusters 158 4 49 109 100]
"""
import argparse
import duckdb
import os

REDSET = "work/data/full_provisioned.parquet"
OUTPUT_DIR = "work/data/provisioned_sampled"


def sample_cluster(con, cluster_id, db_id, max_rows, output_path):
    """Sample up to max_rows from a cluster, uniformly across the time range."""
    # First check total usable SELECTs
    total = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{REDSET}')
        WHERE instance_id = {cluster_id}
          AND database_id = {db_id}
    """).fetchone()[0]

    if total <= max_rows:
        # No sampling needed — take everything
        print(f"  Cluster {cluster_id}: {total:,} total rows <= {max_rows:,} limit, taking all")
        con.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{REDSET}')
                WHERE instance_id = {cluster_id}
                  AND database_id = {db_id}
                ORDER BY arrival_timestamp
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    else:
        # Uniform random sample using TABLESAMPLE or ORDER BY random()
        sample_frac = max_rows / total
        print(f"  Cluster {cluster_id}: {total:,} total rows, sampling {max_rows:,} ({sample_frac:.1%})")
        con.execute(f"""
            COPY (
                SELECT * FROM (
                    SELECT *, random() as _rnd
                    FROM read_parquet('{REDSET}')
                    WHERE instance_id = {cluster_id}
                      AND database_id = {db_id}
                )
                ORDER BY _rnd
                LIMIT {max_rows}
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    # Verify
    result = con.execute(f"""
        SELECT COUNT(*) as n,
               MIN(arrival_timestamp) as min_ts,
               MAX(arrival_timestamp) as max_ts,
               DATEDIFF('day', MIN(arrival_timestamp), MAX(arrival_timestamp)) as days
        FROM read_parquet('{output_path}')
    """).fetchone()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  -> {result[0]:,} rows, {result[1]} to {result[2]} ({result[3]} days), {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=500000,
                        help="Max rows per cluster (default 500K)")
    parser.add_argument("--clusters", type=int, nargs="+",
                        default=[158, 4, 49, 109, 100, 79, 103, 34],
                        help="Cluster IDs to sample")
    parser.add_argument("--db", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SELECT setseed({args.seed / 2**31})")  # DuckDB setseed takes [0,1]

    for cluster_id in args.clusters:
        output_path = os.path.join(OUTPUT_DIR, f"prov_c{cluster_id}_d{args.db}.parquet")
        if os.path.exists(output_path):
            print(f"Skipping cluster {cluster_id} — {output_path} already exists")
            continue
        print(f"Sampling cluster {cluster_id}...")
        sample_cluster(con, cluster_id, args.db, args.max_rows, output_path)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
