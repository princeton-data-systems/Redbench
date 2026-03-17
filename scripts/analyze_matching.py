#!/usr/bin/env python3
"""Analyze TPC-H matching workload temporal variability.

For matching workloads, approximated_scan_selectivities is empty.
Instead, we extract filter columns from the SQL WHERE clauses and
analyze their temporal distribution, comparable to the generation analysis.
"""
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def find_workload(output_dir, cluster_id, db_id, strategy="matching"):
    for name in ["tpch", "imdb"]:
        pattern = os.path.join(output_dir, "generated_workloads", name, "serverless",
                               f"cluster_{cluster_id}", f"database_{db_id}",
                               f"{strategy}_*", "workload.csv")
        paths = glob.glob(pattern)
        if paths:
            return paths[0]
    return None


# TPC-H column aliases → canonical table.column
# These are the WHERE-clause filter columns used by TPC-H templates
TPCH_FILTER_PATTERNS = [
    # lineitem filters
    (r"\bl_shipdate\b", "lineitem", "l_shipdate"),
    (r"\bl_commitdate\b", "lineitem", "l_commitdate"),
    (r"\bl_receiptdate\b", "lineitem", "l_receiptdate"),
    (r"\bl_returnflag\b", "lineitem", "l_returnflag"),
    (r"\bl_linestatus\b", "lineitem", "l_linestatus"),
    (r"\bl_shipmode\b", "lineitem", "l_shipmode"),
    (r"\bl_shipinstruct\b", "lineitem", "l_shipinstruct"),
    (r"\bl_quantity\b", "lineitem", "l_quantity"),
    (r"\bl_discount\b", "lineitem", "l_discount"),
    (r"\bl_extendedprice\b", "lineitem", "l_extendedprice"),
    (r"\bl_partkey\b", "lineitem", "l_partkey"),
    (r"\bl_suppkey\b", "lineitem", "l_suppkey"),
    (r"\bl_orderkey\b", "lineitem", "l_orderkey"),
    # orders filters
    (r"\bo_orderdate\b", "orders", "o_orderdate"),
    (r"\bo_orderpriority\b", "orders", "o_orderpriority"),
    (r"\bo_orderstatus\b", "orders", "o_orderstatus"),
    (r"\bo_custkey\b", "orders", "o_custkey"),
    (r"\bo_orderkey\b", "orders", "o_orderkey"),
    (r"\bo_comment\b", "orders", "o_comment"),
    (r"\bo_totalprice\b", "orders", "o_totalprice"),
    # customer filters
    (r"\bc_mktsegment\b", "customer", "c_mktsegment"),
    (r"\bc_nationkey\b", "customer", "c_nationkey"),
    (r"\bc_custkey\b", "customer", "c_custkey"),
    (r"\bc_acctbal\b", "customer", "c_acctbal"),
    (r"\bc_phone\b", "customer", "c_phone"),
    # part filters
    (r"\bp_brand\b", "part", "p_brand"),
    (r"\bp_type\b", "part", "p_type"),
    (r"\bp_size\b", "part", "p_size"),
    (r"\bp_container\b", "part", "p_container"),
    (r"\bp_name\b", "part", "p_name"),
    (r"\bp_mfgr\b", "part", "p_mfgr"),
    # supplier filters
    (r"\bs_nationkey\b", "supplier", "s_nationkey"),
    (r"\bs_comment\b", "supplier", "s_comment"),
    (r"\bs_suppkey\b", "supplier", "s_suppkey"),
    # nation filters
    (r"\bn_name\b", "nation", "n_name"),
    (r"\bn_nationkey\b", "nation", "n_nationkey"),
    # region filters
    (r"\br_name\b", "region", "r_name"),
    # partsupp filters
    (r"\bps_supplycost\b", "partsupp", "ps_supplycost"),
    (r"\bps_availqty\b", "partsupp", "ps_availqty"),
]


def extract_where_clause(sql):
    """Extract WHERE clause from SQL, handling nested subqueries."""
    sql_lower = sql.lower()
    # Find all WHERE clauses (including in subqueries)
    where_parts = []
    for m in re.finditer(r'\bwhere\b', sql_lower):
        start = m.end()
        # Find the end: GROUP BY, ORDER BY, HAVING, LIMIT, or end of string
        end_match = re.search(r'\b(group\s+by|order\s+by|having|limit)\b', sql_lower[start:])
        if end_match:
            where_parts.append(sql_lower[start:start + end_match.start()])
        else:
            where_parts.append(sql_lower[start:])
    return " ".join(where_parts)


def extract_filter_columns(sql):
    """Extract (table, column) pairs from WHERE clause of a TPC-H query."""
    where = extract_where_clause(sql)
    if not where:
        return []

    found = set()
    for pattern, table, col in TPCH_FILTER_PATTERNS:
        if re.search(pattern, where):
            found.add((table, col))
    return list(found)


def build_filter_df(wl):
    """Build filter DataFrame from SQL WHERE clauses."""
    rows = []
    for _, row in wl.iterrows():
        sql = row.get("sql", "")
        if not isinstance(sql, str):
            continue
        ts = row["ts"]
        filters = extract_filter_columns(sql)
        for table, col in filters:
            rows.append((table, col, ts))
    return pd.DataFrame(rows, columns=["table", "col", "ts"])


def compute_jsd(fdf, time_col, tables):
    """Compute count-based JSD for filter column distributions over time."""
    results = {}
    for table in tables:
        tdf = fdf[fdf["table"] == table]
        cols = sorted(tdf["col"].unique())
        n_cols = len(cols)
        if n_cols < 2:
            continue

        periods = sorted(tdf[time_col].unique())
        if len(periods) < 2:
            continue

        # Build distribution per period
        dists = []
        for period in periods:
            pdf = tdf[tdf[time_col] == period]
            counts = pdf["col"].value_counts()
            dist = np.array([counts.get(c, 0) for c in cols], dtype=float)
            total = dist.sum()
            if total > 0:
                dist /= total
            dists.append(dist)

        # Compute mean distribution
        mean_dist = np.mean(dists, axis=0)

        # Compute JSD of each period vs mean
        jsds = []
        for dist in dists:
            if dist.sum() > 0 and mean_dist.sum() > 0:
                jsds.append(jensenshannon(dist, mean_dist) ** 2)

        total_queries = len(tdf)
        results[table] = {
            "jsd_mean": np.mean(jsds) if jsds else 0,
            "jsd_std": np.std(jsds) if jsds else 0,
            "n_cols": n_cols,
            "total_queries": total_queries,
            "n_periods": len(periods),
        }
    return results


def analyze_template_distribution(wl, out_dir, label):
    """Analyze how TPC-H template usage varies over time."""
    # Extract template from SQL (use read_tables + join count as proxy)
    wl["day"] = wl["ts"].dt.date
    wl["week"] = ((wl["ts"] - wl["ts"].min()).dt.days // 7) + 1

    # Template distribution by read_tables (which tables are accessed)
    daily_tables = wl.groupby(["day", "read_tables"]).size().unstack(fill_value=0)

    lines = []
    lines.append(f"\n  Template/Table Access Distribution Over Time")
    lines.append(f"  Total queries: {len(wl)}")
    lines.append(f"  Date range: {wl['ts'].min().date()} to {wl['ts'].max().date()}")

    # Table access frequency
    table_counts = wl["read_tables"].value_counts()
    lines.append(f"\n  Table access frequency (top 15):")
    for tables, count in table_counts.head(15).items():
        pct = 100 * count / len(wl)
        lines.append(f"    {tables:50s} {count:>7d} ({pct:5.1f}%)")

    return "\n".join(lines)


def plot_filter_timelines(fdf, tables, cluster_label, out_dir):
    """Plot filter column frequency timelines for matching workloads."""
    safe_label = cluster_label.replace("/", "_").replace(" ", "_")

    for table in tables:
        tdf = fdf[fdf["table"] == table].copy()
        if tdf.empty:
            continue

        tdf["day"] = tdf["ts"].dt.date
        daily = tdf.groupby(["day", "col"]).size().unstack(fill_value=0)

        # Normalize to proportions
        daily_pct = daily.div(daily.sum(axis=1), axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{cluster_label} — {table} (Matching)", fontsize=14)

        # Raw counts
        daily.plot.area(ax=ax1, alpha=0.7)
        ax1.set_ylabel("Filter count")
        ax1.set_title("Daily filter column counts")
        ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        # Proportions
        daily_pct.plot.area(ax=ax2, alpha=0.7)
        ax2.set_ylabel("Proportion")
        ax2.set_title("Daily filter column proportions")
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"matching_{safe_label}_{table}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def analyze_matching_cluster(output_dir, cluster_id, db_id, cluster_label, out_dir):
    """Full analysis for one matching workload."""
    wl_path = find_workload(output_dir, cluster_id, db_id, strategy="matching")
    if not wl_path:
        print(f"SKIP {cluster_label}: matching workload not found")
        return None

    print(f"\nAnalyzing {cluster_label} (matching) from {wl_path}...")
    wl = pd.read_csv(wl_path)
    wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")

    lines = []
    lines.append(f"\n{'#' * 80}")
    lines.append(f"  {cluster_label} — Matching Workload Analysis")
    lines.append(f"{'#' * 80}")
    lines.append(f"  Total queries: {len(wl)}")
    lines.append(f"  Date range: {wl['ts'].min().date()} to {wl['ts'].max().date()}")
    lines.append(f"  Queries/day: {len(wl) / max(1, (wl['ts'].max() - wl['ts'].min()).days):.1f}")

    # Template/table distribution
    lines.append(analyze_template_distribution(wl, out_dir, cluster_label))

    # Extract filter columns from SQL
    print(f"  Extracting filter columns from SQL...")
    fdf = build_filter_df(wl)
    print(f"  {len(fdf)} filter observations across {fdf['table'].nunique()} tables")

    if fdf.empty:
        lines.append("\n  No filter columns extracted from SQL.")
        return "\n".join(lines)

    # Add time columns
    min_date = fdf["ts"].min()
    fdf["week"] = ((fdf["ts"] - min_date).dt.days // 7) + 1
    fdf["day"] = fdf["ts"].dt.date

    # Tables with >= 2 filter columns
    table_col_counts = fdf.groupby("table")["col"].nunique()
    interesting_tables = table_col_counts[table_col_counts >= 2].index.tolist()

    # Per-column frequency stats
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  Per-Column Filter Frequency (matching workload)")
    lines.append(f"{'=' * 90}")

    for table in sorted(interesting_tables):
        tdf = fdf[fdf["table"] == table]
        col_counts = tdf["col"].value_counts()
        total = col_counts.sum()
        lines.append(f"\n  {table} — {total} filter observations, {len(col_counts)} columns")
        lines.append(f"  {'col':25s} {'count':>7s} {'pct':>7s}")
        lines.append(f"  {'-'*25} {'-'*7} {'-'*7}")
        for col, count in col_counts.items():
            pct = 100 * count / total
            lines.append(f"  {col:25s} {count:>7d} {pct:>6.1f}%")

    # JSD analysis
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  JSD Analysis: Filter Column Distribution Over Time")
    lines.append(f"{'=' * 90}")

    # Weekly JSD
    weekly = compute_jsd(fdf, "week", interesting_tables)
    lines.append(f"\n  Weekly JSD:")
    lines.append(f"  {'table':20s} {'queries':>8s} {'cols':>5s} {'JSD':>10s}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*10}")
    for table in sorted(weekly, key=lambda t: weekly[t]["jsd_mean"], reverse=True):
        d = weekly[table]
        lines.append(f"  {table:20s} {d['total_queries']:>8d} {d['n_cols']:>5d} {d['jsd_mean']:>10.4f}")

    # Daily JSD
    daily = compute_jsd(fdf, "day", interesting_tables)
    lines.append(f"\n  Daily JSD:")
    lines.append(f"  {'table':20s} {'queries':>8s} {'cols':>5s} {'JSD':>10s}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*10}")
    for table in sorted(daily, key=lambda t: daily[t]["jsd_mean"], reverse=True):
        d = daily[table]
        lines.append(f"  {table:20s} {d['total_queries']:>8d} {d['n_cols']:>5d} {d['jsd_mean']:>10.4f}")

    # Comparison summary
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  Summary: Matching vs Generation Comparison")
    lines.append(f"{'=' * 90}")
    lines.append(f"  (Compare these JSD values with generation-based analysis)")
    lines.append(f"  Matching workloads use real TPC-H template SQL with realistic")
    lines.append(f"  filter patterns, but template assignment is deterministic per QIG,")
    lines.append(f"  so temporal variability depends entirely on QIG composition shifts.")

    # Generate plots
    plot_tables = sorted(interesting_tables,
                        key=lambda t: daily.get(t, {}).get("jsd_mean", 0),
                        reverse=True)[:6]
    plot_filter_timelines(fdf, plot_tables, cluster_label, out_dir)

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tpch"], default="tpch",
                       help="Only TPC-H matching is supported currently")
    args = parser.parse_args()

    out_dir = "work/output/matching_analysis"
    os.makedirs(out_dir, exist_ok=True)

    # Only cluster 104 has matching workload for now
    clusters = [
        ("work/output_tpch_sf1_c104_d0", 104, 0, "TPC-H Matching 104/0"),
    ]

    all_text = []
    for out, cid, did, label in clusters:
        result = analyze_matching_cluster(out, cid, did, label, out_dir)
        if result:
            all_text.append(result)

    report_path = os.path.join(out_dir, "matching_analysis.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(all_text))
    print(f"\nWritten: {report_path}")


if __name__ == "__main__":
    main()
