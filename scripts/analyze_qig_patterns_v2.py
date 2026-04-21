#!/usr/bin/env python3
"""
Focused analysis: why does the column distribution shift every hour?
Look at a specific table (title) on cluster 104 IMDB.
"""

import ast
import glob
import os
import re
from collections import defaultdict, Counter

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


# Load cluster 104 IMDB
pattern = "work/output_c104_d0/generated_workloads/imdb/serverless/cluster_104/database_0/generation_*/workload.csv"
wl_path = glob.glob(pattern)[0]
print(f"Loading {wl_path}...")

wl = pd.read_csv(wl_path, low_memory=False)
wl = wl[wl["query_type"] == "select"].copy()
wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")
wl["hour"] = wl["ts"].dt.floor("h")

qig_col = "structural_repetition_id"

# Extract filter observations for 'title' table
rows = []
for _, row in wl.iterrows():
    sels = parse_sels(row.get("approximated_scan_selectivities", "[]"))
    for t, c, sel in sels:
        if t == "title":
            rows.append((c, sel, row["hour"], row[qig_col]))

tdf = pd.DataFrame(rows, columns=["col", "sel", "hour", "qig"])
print(f"Title table: {len(tdf)} filter observations, {tdf['col'].nunique()} columns")
print(f"Columns: {sorted(tdf['col'].unique())}")

# ── QIG analysis for title table ───────────────────────────────────────
print(f"\n{'='*80}")
print("QIG ANALYSIS FOR 'title' TABLE")
print(f"{'='*80}")

qig_stats = tdf.groupby("qig").agg(
    count=("col", "size"),
    first_hour=("hour", "min"),
    last_hour=("hour", "max"),
    n_hours=("hour", "nunique"),
).reset_index()
qig_stats["lifespan_hours"] = (qig_stats["last_hour"] - qig_stats["first_hour"]).dt.total_seconds() / 3600
qig_stats = qig_stats.sort_values("count", ascending=False)

print(f"\nTotal QIGs touching title: {len(qig_stats)}")
print(f"  Span >30 days: {(qig_stats['lifespan_hours'] > 720).sum()}")
print(f"  Span 7-30 days: {((qig_stats['lifespan_hours'] > 168) & (qig_stats['lifespan_hours'] <= 720)).sum()}")
print(f"  Span 1-7 days: {((qig_stats['lifespan_hours'] > 24) & (qig_stats['lifespan_hours'] <= 168)).sum()}")
print(f"  Span <1 day: {(qig_stats['lifespan_hours'] <= 24).sum()}")

# What column does each QIG produce on title?
print(f"\nTop 20 QIGs for title (each QIG -> which column?):")
for _, row in qig_stats.head(20).iterrows():
    qig_data = tdf[tdf["qig"] == row["qig"]]
    col_counts = qig_data["col"].value_counts()
    col_str = ", ".join(f"{c}={n}" for c, n in col_counts.head(3).items())
    lifespan_d = row["lifespan_hours"] / 24
    print(f"  QIG {row['qig']:>6d}: {row['count']:>5d} obs over {lifespan_d:>5.1f}d "
          f"({row['n_hours']:>4d}h active) -> {col_str}")

# ── Hourly column proportions: show 24 consecutive hours ──────────────
print(f"\n{'='*80}")
print("HOURLY COLUMN PROPORTIONS (24 consecutive hours from middle of dataset)")
print(f"{'='*80}")

all_cols = sorted(tdf["col"].unique())
hours_sorted = sorted(tdf["hour"].unique())
mid = len(hours_sorted) // 2

# Print header
col_header = "  ".join(f"{c[:8]:>8s}" for c in all_cols)
print(f"{'hour':>20s} {'n':>5s}  {col_header}  {'n_qigs':>6s}")

for h in hours_sorted[mid:mid+24]:
    hdf = tdf[tdf["hour"] == h]
    n = len(hdf)
    if n == 0:
        continue
    props = []
    for c in all_cols:
        cnt = (hdf["col"] == c).sum()
        props.append(f"{cnt/n:>8.2f}")
    n_qigs = hdf["qig"].nunique()
    h_str = str(h)[:16]
    print(f"{h_str:>20s} {n:>5d}  {'  '.join(props)}  {n_qigs:>6d}")

# ── The key question: within each hour, how many QIGs, and do they ────
# ── produce different columns? ─────────────────────────────────────────
print(f"\n{'='*80}")
print("WHY DOES THE DISTRIBUTION SHIFT? (detailed hour breakdown)")
print(f"{'='*80}")

for h in hours_sorted[mid:mid+6]:
    hdf = tdf[tdf["hour"] == h]
    n = len(hdf)
    if n == 0:
        continue
    print(f"\n  {str(h)[:16]} — {n} filter obs, {hdf['qig'].nunique()} QIGs")

    # Group by QIG, show what column each produces
    qig_groups = hdf.groupby("qig")["col"].value_counts().reset_index()
    qig_groups.columns = ["qig", "col", "count"]
    qig_totals = qig_groups.groupby("qig")["count"].sum().sort_values(ascending=False)

    for qig in qig_totals.head(8).index:
        qig_data = qig_groups[qig_groups["qig"] == qig]
        total = qig_data["count"].sum()
        cols = ", ".join(f"{r['col']}={r['count']}" for _, r in qig_data.iterrows())
        print(f"    QIG {qig:>6d}: {total:>4d} obs -> {cols}")
    if len(qig_totals) > 8:
        remaining = qig_totals.iloc[8:].sum()
        print(f"    ... +{len(qig_totals)-8} more QIGs with {remaining} obs")

# ── Day-level analysis: are daily distributions more stable? ───────────
print(f"\n{'='*80}")
print("DAILY COLUMN PROPORTIONS (14 consecutive days)")
print(f"{'='*80}")

tdf["day"] = tdf["hour"].dt.date
days_sorted = sorted(tdf["day"].unique())
mid_d = len(days_sorted) // 2

col_header = "  ".join(f"{c[:8]:>8s}" for c in all_cols)
print(f"{'day':>12s} {'n':>5s}  {col_header}")

for d in days_sorted[mid_d:mid_d+14]:
    ddf = tdf[tdf["day"] == d]
    n = len(ddf)
    if n == 0:
        continue
    props = []
    for c in all_cols:
        cnt = (ddf["col"] == c).sum()
        props.append(f"{cnt/n:>8.2f}")
    print(f"{str(d):>12s} {n:>5d}  {'  '.join(props)}")
