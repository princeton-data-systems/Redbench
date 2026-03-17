"""
Analyze filter diversity in full 3-month SELECT-only workloads.
Focus: how filter occurrence and selectivity on specific columns change over time.
"""

import ast, re, os, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import Counter

# Auto-detect workload paths in output_select_full
def find_workload(strategy):
    pattern = f"work/output_select_full/generated_workloads/imdb/serverless/cluster_0/database_0/{strategy}_*/workload.csv"
    paths = glob.glob(pattern)
    if not paths:
        return None
    return paths[0]

OUTPUT_DIR = "work/output_select_full/filter_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parsing helpers (same as before) ---

def parse_generation_filters(row):
    raw = row["approximated_scan_selectivities"]
    if pd.isna(raw) or raw == "[]":
        return []
    tuples = ast.literal_eval(raw)
    return [(re.sub(r"_\d+$", "", t), col, sel) for t, col, sel in tuples]

def parse_matching_filters(row):
    sql = row["sql"]
    if pd.isna(sql):
        return []
    where_match = re.search(r"\bWHERE\b(.+)", sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return []
    where_clause = where_match.group(1)
    conditions = re.split(r"\bAND\b", where_clause, flags=re.IGNORECASE)
    filters = []
    join_pat = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*;?\s*$", re.IGNORECASE)
    filter_pat = re.compile(r"(\w+)\.(\w+)\s*(?:LIKE|IN|BETWEEN|>=?|<=?|!=|=)\s*", re.IGNORECASE)
    for cond in conditions:
        cond = cond.strip().rstrip(";")
        if join_pat.match(cond):
            continue
        m = filter_pat.match(cond)
        if m:
            filters.append((m.group(1), m.group(2), None))
    return filters

def load_selects(path):
    df = pd.read_csv(path)
    df = df[df["query_type"] == "select"].copy()
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    return df.sort_values("arrival_timestamp")

# --- Per-column temporal analysis ---

def plot_per_column_over_time(df, label, parse_fn, output_prefix):
    """
    For the top N most-used filter columns, plot:
      1. Occurrence count per week
      2. Mean selectivity per week (generation only)
    """
    df = df.copy()
    df["filters"] = df.apply(parse_fn, axis=1)

    # Flatten to per-filter rows
    records = []
    for _, row in df.iterrows():
        ts = row["arrival_timestamp"]
        for t, c, sel in row["filters"]:
            records.append({"timestamp": ts, "table": t, "col": c, "selectivity": sel})

    if not records:
        print(f"  No filters found for {label}")
        return

    fdf = pd.DataFrame(records)
    fdf["week"] = fdf["timestamp"].dt.to_period("W").apply(lambda p: p.start_time)
    fdf["table_col"] = fdf["table"] + "." + fdf["col"]

    # Top columns by total occurrence
    top_cols = fdf["table_col"].value_counts().head(12).index.tolist()
    top_fdf = fdf[fdf["table_col"].isin(top_cols)]

    # --- Plot 1: occurrence count per week per column ---
    weekly_counts = top_fdf.groupby(["week", "table_col"]).size().unstack(fill_value=0)
    # Reindex to ensure all weeks present
    all_weeks = pd.date_range(fdf["week"].min(), fdf["week"].max(), freq="W-MON")
    weekly_counts = weekly_counts.reindex(all_weeks, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    for col_name in top_cols:
        if col_name in weekly_counts.columns:
            ax.plot(weekly_counts.index, weekly_counts[col_name], marker=".", label=col_name, linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Filter occurrences")
    ax.set_title(f"{label}: filter column occurrence over time (top 12)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_col_occurrence.png"), dpi=150)
    plt.close()
    print(f"  Saved {output_prefix}_col_occurrence.png")

    # --- Plot 2: selectivity per week per column (generation only) ---
    has_sel = fdf["selectivity"].notna().any()
    if has_sel:
        sel_fdf = top_fdf[top_fdf["selectivity"].notna()]
        if len(sel_fdf) > 0:
            weekly_sel = sel_fdf.groupby(["week", "table_col"])["selectivity"].mean().unstack()
            weekly_sel = weekly_sel.reindex(all_weeks)

            fig, ax = plt.subplots(figsize=(14, 6))
            for col_name in top_cols:
                if col_name in weekly_sel.columns:
                    series = weekly_sel[col_name].dropna()
                    if len(series) > 0:
                        ax.plot(series.index, series.values, marker=".", label=col_name, linewidth=1.5)
            ax.set_xlabel("Week")
            ax.set_ylabel("Mean selectivity")
            ax.set_title(f"{label}: filter selectivity over time (top 12)")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_col_selectivity.png"), dpi=150)
            plt.close()
            print(f"  Saved {output_prefix}_col_selectivity.png")


def plot_select_timeline(df, label, output_prefix):
    """Scatter plot of distinct SELECT queries over time."""
    df = df.copy()
    # Use exact_repetition_hash to identify distinct queries
    hashes = df["exact_repetition_hash"].unique()
    hash_to_id = {h: i for i, h in enumerate(sorted(hashes, key=lambda h: df[df["exact_repetition_hash"]==h]["arrival_timestamp"].min()))}
    df["query_id"] = df["exact_repetition_hash"].map(hash_to_id)

    n_distinct = len(hashes)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df["arrival_timestamp"], df["query_id"], alpha=0.3, s=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Query ID (ordered by first appearance)")
    ax.set_title(f"{label}: SELECT query execution timeline ({n_distinct} distinct queries, {len(df)} total)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_timeline.png"), dpi=150)
    plt.close()
    print(f"  Saved {output_prefix}_timeline.png")


def summarize(df, label, parse_fn):
    df = df.copy()
    df["filters"] = df.apply(parse_fn, axis=1)
    df["num_filter_cols"] = df["filters"].apply(len)

    all_cols = set()
    all_sels = []
    col_counts = Counter()
    for flist in df["filters"]:
        for t, c, s in flist:
            all_cols.add((t, c))
            col_counts[(t, c)] += 1
            if s is not None:
                all_sels.append(s)

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(df)} SELECT queries)")
    print(f"{'='*60}")
    print(f"  Distinct (table, column) pairs filtered: {len(all_cols)}")
    print(f"  Filter predicates per query: mean={df['num_filter_cols'].mean():.2f}, median={df['num_filter_cols'].median():.1f}, max={df['num_filter_cols'].max()}")
    if all_sels:
        sels = np.array(all_sels)
        print(f"  Selectivity: mean={sels.mean():.3f}, median={np.median(sels):.3f}, min={sels.min():.3f}, max={sels.max():.3f}")
    else:
        print("  Selectivity: not available")
    print(f"  Top 10 filtered columns:")
    for (t, c), cnt in col_counts.most_common(10):
        print(f"    {t}.{c}: {cnt}")


# --- Main ---

matching_path = find_workload("matching")
generation_path = find_workload("generation")

if matching_path:
    print(f"Matching: {matching_path}")
    match_df = load_selects(matching_path)
    summarize(match_df, "Matching", parse_matching_filters)
    plot_select_timeline(match_df, "Matching", "matching")
    plot_per_column_over_time(match_df, "Matching", parse_matching_filters, "matching")
else:
    print("Matching workload not found yet")

if generation_path:
    print(f"\nGeneration: {generation_path}")
    gen_df = load_selects(generation_path)
    summarize(gen_df, "Generation", parse_generation_filters)
    plot_select_timeline(gen_df, "Generation", "generation")
    plot_per_column_over_time(gen_df, "Generation", parse_generation_filters, "generation")
else:
    print("Generation workload not found yet")
