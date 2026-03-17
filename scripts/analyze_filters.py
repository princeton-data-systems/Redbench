"""
Analyze filter diversity in generated (generation) and matched (matching) SELECT workloads.

Metrics:
  1. Number of distinct (table, column) pairs filtered on
  2. Distribution of selectivities per filtered column
  3. How these change over time (by arrival timestamp)
"""

import ast
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

MATCHING_PATH = "output/generated_workloads/imdb/serverless/cluster_0/database_0/matching_8a496730ead67c73b559bd05c605c1c4/workload.csv"
GENERATION_PATH = "output/generated_workloads/imdb/serverless/cluster_0/database_0/generation_ede5387599ee1e65c105eaa9b17c5c3c/workload.csv"
OUTPUT_DIR = "output/filter_analysis"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_generation_filters(row):
    """Parse the structured approximated_scan_selectivities column."""
    raw = row["approximated_scan_selectivities"]
    if pd.isna(raw) or raw == "[]":
        return []
    tuples = ast.literal_eval(raw)
    # strip version suffix from table names (e.g. kind_type_0 -> kind_type)
    return [(re.sub(r"_\d+$", "", t), col, sel) for t, col, sel in tuples]


def parse_matching_filters(row):
    """
    Extract filter columns from matching SQL using simple regex.
    We look for non-join WHERE predicates: column LIKE/IN/>/</>=/<=/=/BETWEEN patterns.
    Join predicates (table.col = table.col) are excluded.
    No selectivity info is available for matching.
    """
    sql = row["sql"]
    if pd.isna(sql):
        return []

    # Find WHERE clause
    where_match = re.search(r"\bWHERE\b(.+)", sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return []
    where_clause = where_match.group(1)

    # Split on AND
    conditions = re.split(r"\bAND\b", where_clause, flags=re.IGNORECASE)

    filters = []
    # Join pattern: alias.col = alias.col
    join_pat = re.compile(
        r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*;?\s*$", re.IGNORECASE
    )
    # Filter pattern: alias.col <op> <value>
    filter_pat = re.compile(
        r"(\w+)\.(\w+)\s*(?:LIKE|IN|BETWEEN|>=?|<=?|!=|=)\s*", re.IGNORECASE
    )

    for cond in conditions:
        cond = cond.strip().rstrip(";")
        if join_pat.match(cond):
            continue
        m = filter_pat.match(cond)
        if m:
            table = m.group(1)
            col = m.group(2)
            filters.append((table, col, None))  # no selectivity for matching
    return filters


# ---------------------------------------------------------------------------
# Load and extract
# ---------------------------------------------------------------------------

def load_selects(path):
    df = pd.read_csv(path)
    df = df[df["query_type"] == "select"].copy()
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    return df


gen_df = load_selects(GENERATION_PATH)
match_df = load_selects(MATCHING_PATH)

gen_df["filters"] = gen_df.apply(parse_generation_filters, axis=1)
match_df["filters"] = match_df.apply(parse_matching_filters, axis=1)

gen_df["num_filter_cols"] = gen_df["filters"].apply(len)
match_df["num_filter_cols"] = match_df["filters"].apply(len)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def summarize(df, label):
    all_cols = set()
    all_sels = []
    for flist in df["filters"]:
        for t, c, s in flist:
            all_cols.add((t, c))
            if s is not None:
                all_sels.append(s)

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(df)} SELECT queries)")
    print(f"{'='*60}")
    print(f"  Distinct (table, column) pairs filtered: {len(all_cols)}")
    print(f"  Filter predicates per query: "
          f"mean={df['num_filter_cols'].mean():.2f}, "
          f"median={df['num_filter_cols'].median():.1f}, "
          f"max={df['num_filter_cols'].max()}")
    if all_sels:
        sels = np.array(all_sels)
        print(f"  Selectivity: mean={sels.mean():.3f}, "
              f"median={np.median(sels):.3f}, "
              f"min={sels.min():.3f}, max={sels.max():.3f}")
    else:
        print("  Selectivity: not available")
    print(f"  Filtered columns (top 10):")
    from collections import Counter
    col_counts = Counter()
    for flist in df["filters"]:
        for t, c, _ in flist:
            col_counts[(t, c)] += 1
    for (t, c), cnt in col_counts.most_common(10):
        print(f"    {t}.{c}: {cnt}")


summarize(gen_df, "Generation")
summarize(match_df, "Matching")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 1. Filter count per query (histogram) ---
ax = axes[0, 0]
bins = range(0, max(gen_df["num_filter_cols"].max(), match_df["num_filter_cols"].max()) + 2)
ax.hist(gen_df["num_filter_cols"], bins=bins, alpha=0.6, label="Generation", edgecolor="black")
ax.hist(match_df["num_filter_cols"], bins=bins, alpha=0.6, label="Matching", edgecolor="black")
ax.set_xlabel("# filter predicates per query")
ax.set_ylabel("Count")
ax.set_title("Filter predicates per SELECT query")
ax.legend()

# --- 2. Selectivity distribution (generation only) ---
ax = axes[0, 1]
gen_sels = [s for flist in gen_df["filters"] for _, _, s in flist if s is not None]
if gen_sels:
    ax.hist(gen_sels, bins=20, alpha=0.7, edgecolor="black", color="tab:blue")
ax.set_xlabel("Selectivity")
ax.set_ylabel("Count")
ax.set_title("Filter selectivity distribution (Generation)")

# --- 3. Distinct filtered columns over time (cumulative) ---
ax = axes[1, 0]
for df, label, color in [(gen_df, "Generation", "tab:blue"), (match_df, "Matching", "tab:orange")]:
    sorted_df = df.sort_values("arrival_timestamp")
    seen = set()
    cum = []
    times = []
    for _, row in sorted_df.iterrows():
        for t, c, _ in row["filters"]:
            seen.add((t, c))
        cum.append(len(seen))
        times.append(row["arrival_timestamp"])
    ax.plot(times, cum, label=label, color=color)
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative distinct (table, col) pairs")
ax.set_title("Filter column diversity over time")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.tick_params(axis="x", rotation=30)

# --- 4. Filter count over time (rolling average) ---
ax = axes[1, 1]
for df, label, color in [(gen_df, "Generation", "tab:blue"), (match_df, "Matching", "tab:orange")]:
    sorted_df = df.sort_values("arrival_timestamp")
    window = max(1, len(sorted_df) // 5)
    rolling = sorted_df["num_filter_cols"].rolling(window, min_periods=1).mean()
    ax.plot(sorted_df["arrival_timestamp"].values, rolling.values, label=label, color=color)
ax.set_xlabel("Time")
ax.set_ylabel("Avg # filter predicates (rolling)")
ax.set_title("Filter count over time")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "filter_diversity.png"), dpi=150)
print(f"\nPlot saved to {OUTPUT_DIR}/filter_diversity.png")
