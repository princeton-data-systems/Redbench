#!/usr/bin/env python3
"""
Analyze QIG temporal patterns in Redbench workloads.

Questions:
1. How many QIGs are there, and how long does each one span?
2. Are QIGs bursty (concentrated in short periods) or spread across the full 3 months?
3. What does the hourly QIG composition look like — is it stable or shifting?
4. Why does the column distribution shift every hour?
"""

import ast
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def strip_suffix(table):
    if table and table[-1].isdigit() and "_" in table:
        return table.rsplit("_", 1)[0]
    return table


def parse_sels(s):
    try:
        items = ast.literal_eval(s) if isinstance(s, str) else []
        return [(strip_suffix(t), c, float(sel)) for t, c, sel in items]
    except Exception:
        return []


def find_workload(output_dir, cluster_id, db_id):
    for db_name in ["imdb", "tpch"]:
        for redset_type in ["serverless", "provisioned"]:
            pattern = os.path.join(output_dir, "generated_workloads", db_name, redset_type,
                                   f"cluster_{cluster_id}", f"database_{db_id}",
                                   "generation_*", "workload.csv")
            paths = glob.glob(pattern)
            if paths:
                return paths[0], db_name
    return None, None


def analyze_cluster(output_dir, cluster_id, db_id, label):
    wl_path, db_name = find_workload(output_dir, cluster_id, db_id)
    if not wl_path:
        print(f"SKIP {label}")
        return

    print(f"\n{'='*80}")
    print(f"  {label} — {wl_path}")
    print(f"{'='*80}")

    wl = pd.read_csv(wl_path, low_memory=False)
    wl = wl[wl["query_type"] == "select"].copy()
    wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")
    wl["hour"] = wl["ts"].dt.floor("h")
    wl["day"] = wl["ts"].dt.date

    # Use structural_repetition_id as QIG proxy (or query_hash)
    # The report says QIG = (query_type, num_joins, num_aggregations, read_table_ids, ...)
    # structural_repetition_id should capture this
    qig_col = "structural_repetition_id" if "structural_repetition_id" in wl.columns else "query_hash"
    print(f"  Using '{qig_col}' as QIG identifier")
    print(f"  Total SELECT queries: {len(wl)}")

    # Extract filter info per query
    filter_cols_per_query = []
    for _, row in wl.iterrows():
        sels = parse_sels(row.get("approximated_scan_selectivities", "[]"))
        # Get the set of (table, col) for this query
        cols = [(t, c) for t, c, s in sels]
        filter_cols_per_query.append(cols)
    wl["filter_cols"] = filter_cols_per_query

    # ── QIG lifespan analysis ──────────────────────────────────────────
    qig_stats = wl.groupby(qig_col).agg(
        count=("ts", "size"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        n_days=("day", "nunique"),
        n_hours=("hour", "nunique"),
    ).reset_index()
    qig_stats["lifespan_days"] = (qig_stats["last_ts"] - qig_stats["first_ts"]).dt.total_seconds() / 86400
    qig_stats = qig_stats.sort_values("count", ascending=False)

    total_days = (wl["ts"].max() - wl["ts"].min()).total_seconds() / 86400

    print(f"\n  Total QIGs: {len(qig_stats)}")
    print(f"  Time span: {total_days:.1f} days")
    print(f"\n  QIG lifespan distribution:")
    print(f"    Span full period (>80 days): {(qig_stats['lifespan_days'] > 80).sum()}")
    print(f"    Span >50% (>45 days):        {(qig_stats['lifespan_days'] > 45).sum()}")
    print(f"    Span <7 days:                {(qig_stats['lifespan_days'] < 7).sum()}")
    print(f"    Span <1 day:                 {(qig_stats['lifespan_days'] < 1).sum()}")

    print(f"\n  Top 15 QIGs by query count:")
    print(f"    {'QIG':>8s} {'count':>7s} {'%total':>7s} {'lifespan':>9s} {'days_active':>12s} "
          f"{'q/active_day':>12s} {'filter_cols':>30s}")
    for _, row in qig_stats.head(15).iterrows():
        qig = str(row[qig_col])[:8]
        pct = 100 * row["count"] / len(wl)
        qpd = row["count"] / max(1, row["n_days"])
        # Get the filter columns this QIG produces
        qig_queries = wl[wl[qig_col] == row[qig_col]]
        all_filter_cols = set()
        for fc_list in qig_queries["filter_cols"].values[:10]:  # sample
            for tc in fc_list:
                all_filter_cols.add(tc)
        fc_str = ", ".join(f"{t}.{c}" for t, c in sorted(all_filter_cols)[:3])
        if len(all_filter_cols) > 3:
            fc_str += f" +{len(all_filter_cols)-3}"
        print(f"    {qig:>8s} {row['count']:>7d} {pct:>6.1f}% {row['lifespan_days']:>8.1f}d "
              f"{row['n_days']:>11d}d {qpd:>11.1f} {fc_str:>30s}")

    # ── Does each QIG always produce the same filter column? ───────────
    print(f"\n  Filter column consistency per QIG (top 20 QIGs):")
    for _, row in qig_stats.head(20).iterrows():
        qig_queries = wl[wl[qig_col] == row[qig_col]]
        col_sets = defaultdict(int)
        for fc_list in qig_queries["filter_cols"].values:
            key = tuple(sorted(fc_list))
            col_sets[key] += 1
        n_distinct = len(col_sets)
        top_set, top_count = max(col_sets.items(), key=lambda x: x[1])
        consistency = top_count / row["count"]
        top_str = ", ".join(f"{t}.{c}" for t, c in top_set[:3])
        print(f"    QIG {str(row[qig_col])[:8]}: {n_distinct} distinct filter sets, "
              f"top={top_str} ({consistency:.0%} of queries)")

    # ── Hourly composition: how many QIGs active per hour? ─────────────
    hourly_qig_counts = wl.groupby("hour")[qig_col].nunique()
    hourly_query_counts = wl.groupby("hour").size()

    print(f"\n  Hourly QIG diversity:")
    print(f"    Distinct QIGs per hour: mean={hourly_qig_counts.mean():.1f}, "
          f"median={hourly_qig_counts.median():.1f}, "
          f"min={hourly_qig_counts.min()}, max={hourly_qig_counts.max()}")
    print(f"    Queries per hour: mean={hourly_query_counts.mean():.1f}, "
          f"median={hourly_query_counts.median():.1f}")

    # ── Focus on one table: how does the column distribution shift? ────
    # Pick the table with most filter observations
    all_filter_obs = []
    for _, row in wl.iterrows():
        for t, c in row["filter_cols"]:
            all_filter_obs.append((t, c, row["hour"], row[qig_col]))
    fdf = pd.DataFrame(all_filter_obs, columns=["table", "col", "hour", "qig"])

    table_counts = fdf.groupby("table").size().sort_values(ascending=False)
    if len(table_counts) == 0:
        return

    focus_table = table_counts.index[0]
    tdf = fdf[fdf["table"] == focus_table]
    print(f"\n  Focus table: {focus_table} ({len(tdf)} filter observations)")

    # How many distinct columns per hour on this table?
    hourly_col_diversity = tdf.groupby("hour")["col"].nunique()
    hourly_col_counts = tdf.groupby("hour").size()
    print(f"    Distinct cols per hour: mean={hourly_col_diversity.mean():.1f}, "
          f"median={hourly_col_diversity.median():.1f}")
    print(f"    Filter obs per hour: mean={hourly_col_counts.mean():.1f}")

    # Show a few consecutive hours to see what's happening
    all_cols = sorted(tdf["col"].unique())
    hours_sorted = sorted(tdf["hour"].unique())
    print(f"\n    Sample of consecutive hours (column proportions):")
    print(f"    {'hour':>20s} {'n_obs':>6s}  " + "  ".join(f"{c[:10]:>10s}" for c in all_cols[:8]))

    # Pick a stretch of hours from the middle of the dataset
    mid = len(hours_sorted) // 2
    sample_hours = hours_sorted[mid:mid+12]
    for h in sample_hours:
        hdf = tdf[tdf["hour"] == h]
        n = len(hdf)
        props = []
        for c in all_cols[:8]:
            cnt = (hdf["col"] == c).sum()
            props.append(f"{cnt/n:>10.2f}" if n > 0 else f"{'':>10s}")
        h_str = str(h)[:16]
        print(f"    {h_str:>20s} {n:>6d}  " + "  ".join(props))

    # ── Key question: within a single hour, how many QIGs contribute? ──
    print(f"\n    QIGs contributing to {focus_table} per hour:")
    hourly_table_qigs = tdf.groupby("hour")["qig"].nunique()
    print(f"      mean={hourly_table_qigs.mean():.1f}, median={hourly_table_qigs.median():.1f}, "
          f"min={hourly_table_qigs.min()}, max={hourly_table_qigs.max()}")

    # Show the QIG breakdown for a few hours
    print(f"\n    QIG breakdown for sample hours on {focus_table}:")
    for h in sample_hours[:6]:
        hdf = tdf[tdf["hour"] == h]
        qig_breakdown = hdf.groupby("qig")["col"].apply(lambda x: dict(x.value_counts())).to_dict()
        print(f"      {str(h)[:16]}:")
        for qig, col_counts in sorted(qig_breakdown.items(), key=lambda x: -sum(x[1].values()))[:5]:
            total = sum(col_counts.values())
            cols_str = ", ".join(f"{c}={n}" for c, n in sorted(col_counts.items(), key=lambda x: -x[1])[:3])
            print(f"        QIG {str(qig)[:8]}: {total} obs -> {cols_str}")


def main():
    # Focus on the most interesting clusters
    clusters = [
        ("work/output_c104_d0", 104, 0, "IMDB srvl-104"),
        ("work/output_tpch_sf1_c104_d0", 104, 0, "TPC-H srvl-104"),
        ("work/output_c85_d0", 85, 0, "IMDB srvl-85"),
    ]

    for output_dir, cid, did, label in clusters:
        analyze_cluster(output_dir, cid, did, label)


if __name__ == "__main__":
    main()
