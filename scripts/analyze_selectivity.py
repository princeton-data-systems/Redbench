#!/usr/bin/env python3
"""
Selectivity-aware filter analysis for Redbench generated workloads.

For each (table, column) in each time window, computes:
  - count: number of queries filtering on this column
  - mean_selectivity: average selectivity across those queries
  - pruning_weight: sum(1 - selectivity) — total "pruning benefit"

Then measures temporal variation using JSD on pruning-weight distributions
(instead of raw count distributions), and produces per-table summary stats
and stacked area plots showing how the pruning-weighted importance of each
column shifts over time.
"""
import ast
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def parse_sels(s):
    try:
        items = ast.literal_eval(s) if isinstance(s, str) else []
        return [(t, c, float(sel)) for t, c, sel in items]
    except Exception:
        return []


def find_gen_workload(output_dir, cluster_id, db_id, db_name=None, strategy="generation"):
    if db_name:
        db_names = [db_name]
    else:
        db_names = ["imdb", "tpch"]
    for name in db_names:
        for redset_type in ["serverless", "provisioned"]:
            pattern = os.path.join(output_dir, "generated_workloads", name, redset_type,
                                   f"cluster_{cluster_id}", f"database_{db_id}",
                                   f"{strategy}_*", "workload.csv")
            paths = glob.glob(pattern)
            if paths:
                return paths[0]
    return None


def strip_suffix(table):
    """Remove _0 / _1 augmentation suffix."""
    if table[-1].isdigit() and "_" in table:
        return table.rsplit("_", 1)[0]
    return table


def build_filter_df(wl):
    """Extract per-filter rows with (table, col, selectivity, timestamp)."""
    rows = []
    for _, row in wl.iterrows():
        sels = parse_sels(row.get("approximated_scan_selectivities", "[]"))
        ts = row["ts"]
        for table, col, sel in sels:
            rows.append((strip_suffix(table), col, sel, ts))
    return pd.DataFrame(rows, columns=["table", "col", "selectivity", "ts"])


def compute_selectivity_stats(fdf, tables):
    """Per-table, per-column aggregate selectivity statistics."""
    results = []
    for table in tables:
        tdf = fdf[fdf["table"] == table]
        if tdf.empty:
            continue
        for col, cdf in tdf.groupby("col"):
            results.append({
                "table": table,
                "col": col,
                "count": len(cdf),
                "mean_sel": cdf["selectivity"].mean(),
                "median_sel": cdf["selectivity"].median(),
                "p25_sel": cdf["selectivity"].quantile(0.25),
                "p75_sel": cdf["selectivity"].quantile(0.75),
                "min_sel": cdf["selectivity"].min(),
                "max_sel": cdf["selectivity"].max(),
                "pruning_weight": (1 - cdf["selectivity"]).sum(),
            })
    return pd.DataFrame(results)


def compute_weighted_jsd(fdf, time_col, tables):
    """
    For each table, compute pruning-weight distribution per time window,
    then JSD between consecutive windows.
    Also compute count-based JSD for comparison.
    """
    fdf = fdf.copy()
    fdf["pruning"] = 1.0 - fdf["selectivity"]

    results = {}
    for table in tables:
        tdf = fdf[fdf["table"] == table]
        if tdf.empty:
            continue
        all_cols = sorted(tdf["col"].unique())
        if len(all_cols) < 2:
            continue
        windows = sorted(tdf[time_col].unique())
        if len(windows) < 2:
            continue

        # Build distributions per window: count-based and pruning-weighted
        count_dists = []
        weight_dists = []
        window_labels = []
        for w in windows:
            wdf = tdf[tdf[time_col] == w]
            counts = np.array([float((wdf["col"] == c).sum()) for c in all_cols])
            weights = np.array([wdf.loc[wdf["col"] == c, "pruning"].sum() for c in all_cols])
            ct = counts.sum()
            wt = weights.sum()
            if ct > 0 and wt > 0:
                count_dists.append(counts / ct)
                weight_dists.append(weights / wt)
                window_labels.append(w)

        if len(count_dists) < 2:
            continue

        # JSD series
        count_jsds = []
        weight_jsds = []
        for i in range(1, len(count_dists)):
            cj = jensenshannon(count_dists[i - 1], count_dists[i])
            wj = jensenshannon(weight_dists[i - 1], weight_dists[i])
            if not np.isnan(cj):
                count_jsds.append(cj)
            if not np.isnan(wj):
                weight_jsds.append(wj)

        if not weight_jsds:
            continue

        results[table] = {
            "columns": all_cols,
            "n_cols": len(all_cols),
            "total_queries": len(tdf),
            "count_jsd_mean": np.mean(count_jsds) if count_jsds else 0,
            "weight_jsd_mean": np.mean(weight_jsds),
            "weight_jsd_median": np.median(weight_jsds),
            "weight_jsd_p75": np.percentile(weight_jsds, 75),
            "weight_jsd_max": np.max(weight_jsds),
            "n_windows": len(window_labels),
            "window_labels": window_labels,
            "count_dists": count_dists,
            "weight_dists": weight_dists,
        }
    return results


def plot_selectivity_timelines(fdf, tables, cluster_label, out_dir):
    """
    For each table, produce a 3-panel plot:
      Top: stacked area of pruning-weighted column proportions (weekly)
      Middle: stacked area of count-based column proportions (weekly)
      Bottom: per-column mean selectivity over time
    """
    fdf = fdf.copy()
    fdf["pruning"] = 1.0 - fdf["selectivity"]
    min_date = fdf["ts"].min()
    fdf["week"] = ((fdf["ts"] - min_date).dt.days // 7) + 1

    for table in tables:
        tdf = fdf[fdf["table"] == table]
        if len(tdf) < 10:
            continue
        all_cols = sorted(tdf["col"].unique())
        weeks = sorted(tdf["week"].unique())

        # Build matrices
        count_props = np.zeros((len(weeks), len(all_cols)))
        weight_props = np.zeros((len(weeks), len(all_cols)))
        mean_sels = np.full((len(weeks), len(all_cols)), np.nan)
        counts_per_week = np.zeros(len(weeks))

        for i, w in enumerate(weeks):
            wdf = tdf[tdf["week"] == w]
            counts_per_week[i] = len(wdf)
            for j, c in enumerate(all_cols):
                cdf = wdf[wdf["col"] == c]
                count_props[i, j] = len(cdf)
                weight_props[i, j] = cdf["pruning"].sum()
                if len(cdf) > 0:
                    mean_sels[i, j] = cdf["selectivity"].mean()

            ct = count_props[i].sum()
            wt = weight_props[i].sum()
            if ct > 0:
                count_props[i] /= ct
            if wt > 0:
                weight_props[i] /= wt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 3, 2])
        fig.suptitle(f"{cluster_label} — {table} selectivity-aware analysis",
                     fontsize=13, fontweight="bold")

        # Panel 1: pruning-weighted proportions
        axes[0].stackplot(weeks, weight_props.T, labels=all_cols, alpha=0.8)
        axes[0].set_ylabel("Pruning-weighted\nproportion")
        axes[0].set_ylim(0, 1)
        axes[0].legend(loc="upper left", fontsize=7, ncol=min(5, len(all_cols)))
        axes[0].set_xlim(weeks[0], weeks[-1])
        axes[0].set_title("Pruning-weight distribution: sum(1 - selectivity) per column", fontsize=10)

        # Panel 2: count-based proportions
        axes[1].stackplot(weeks, count_props.T, labels=all_cols, alpha=0.8)
        axes[1].set_ylabel("Count-based\nproportion")
        axes[1].set_ylim(0, 1)
        axes[1].set_xlim(weeks[0], weeks[-1])
        axes[1].set_title("Count distribution: number of queries per column", fontsize=10)

        # Panel 3: mean selectivity per column over time
        for j, c in enumerate(all_cols):
            valid = ~np.isnan(mean_sels[:, j])
            if valid.any():
                axes[2].plot(np.array(weeks)[valid], mean_sels[:, j][valid],
                           marker=".", markersize=3, label=c, alpha=0.8)
        axes[2].set_ylabel("Mean selectivity")
        axes[2].set_xlabel("Week (relative)")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlim(weeks[0], weeks[-1])
        axes[2].legend(loc="upper left", fontsize=7, ncol=min(5, len(all_cols)))
        axes[2].set_title("Per-column mean selectivity over time", fontsize=10)
        axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        safe = cluster_label.lower().replace(" ", "_").replace("/", "_")
        fname = os.path.join(out_dir, f"{safe}_{table}_selectivity.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


def analyze_cluster(output_dir, cluster_id, db_id, cluster_label, out_dir, strategy="generation"):
    """Full selectivity-aware analysis for one cluster."""
    wl_path = find_gen_workload(output_dir, cluster_id, db_id, strategy=strategy)
    if not wl_path:
        print(f"SKIP {cluster_label}: workload not found")
        return None

    print(f"\nAnalyzing {cluster_label} from {wl_path}...")
    wl = pd.read_csv(wl_path)
    wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")

    fdf = build_filter_df(wl)
    print(f"  {len(fdf)} filter observations across {fdf['table'].nunique()} tables")

    # Add time columns
    min_date = fdf["ts"].min()
    fdf["week"] = ((fdf["ts"] - min_date).dt.days // 7) + 1
    fdf["day"] = fdf["ts"].dt.date

    # Get tables with >= 2 filter columns
    table_col_counts = fdf.groupby("table")["col"].nunique()
    interesting_tables = table_col_counts[table_col_counts >= 2].index.tolist()

    # Per-column selectivity stats
    stats = compute_selectivity_stats(fdf, interesting_tables)

    lines = []
    lines.append(f"\n{'#' * 80}")
    lines.append(f"  {cluster_label} — Selectivity-Aware Analysis")
    lines.append(f"{'#' * 80}")

    # Per-table summary
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  Per-Column Selectivity Summary (sorted by pruning_weight)")
    lines.append(f"{'=' * 90}")

    for table in sorted(interesting_tables):
        tstat = stats[stats["table"] == table].sort_values("pruning_weight", ascending=False)
        if tstat.empty:
            continue
        total_queries = tstat["count"].sum()
        total_pruning = tstat["pruning_weight"].sum()
        lines.append(f"\n  {table} — {total_queries} filter observations, "
                     f"total pruning weight = {total_pruning:.1f}")
        lines.append(f"  {'col':20s} {'count':>7s} {'mean_sel':>9s} {'med_sel':>8s} "
                     f"{'p25_sel':>8s} {'p75_sel':>8s} {'prune_wt':>9s} {'%_of_total':>10s}")
        lines.append(f"  {'-'*20} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
        for _, r in tstat.iterrows():
            pct = 100 * r["pruning_weight"] / total_pruning if total_pruning > 0 else 0
            lines.append(f"  {r['col']:20s} {r['count']:>7d} {r['mean_sel']:>9.3f} "
                        f"{r['median_sel']:>8.3f} {r['p25_sel']:>8.3f} {r['p75_sel']:>8.3f} "
                        f"{r['pruning_weight']:>9.1f} {pct:>9.1f}%")

    # JSD comparison: count-based vs pruning-weighted
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  JSD Comparison: Count-Based vs Pruning-Weighted (Weekly)")
    lines.append(f"{'=' * 90}")
    lines.append(f"  {'table':20s} {'queries':>8s} {'cols':>5s} "
                 f"{'count_JSD':>10s} {'weight_JSD':>11s} {'delta':>8s}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*10} {'-'*11} {'-'*8}")

    weekly_results = compute_weighted_jsd(fdf, "week", interesting_tables)
    scored = []
    for table, data in weekly_results.items():
        scored.append((data["weight_jsd_mean"], table, data))
    scored.sort(reverse=True)

    for _, table, data in scored:
        delta = data["weight_jsd_mean"] - data["count_jsd_mean"]
        lines.append(f"  {table:20s} {data['total_queries']:>8d} {data['n_cols']:>5d} "
                    f"{data['count_jsd_mean']:>10.4f} {data['weight_jsd_mean']:>11.4f} "
                    f"{delta:>+8.4f}")

    # Daily JSD
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  JSD Comparison: Count-Based vs Pruning-Weighted (Daily)")
    lines.append(f"{'=' * 90}")
    lines.append(f"  {'table':20s} {'queries':>8s} {'cols':>5s} "
                 f"{'count_JSD':>10s} {'weight_JSD':>11s} {'delta':>8s}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*10} {'-'*11} {'-'*8}")

    daily_results = compute_weighted_jsd(fdf, "day", interesting_tables)
    scored_d = []
    for table, data in daily_results.items():
        scored_d.append((data["weight_jsd_mean"], table, data))
    scored_d.sort(reverse=True)

    for _, table, data in scored_d:
        delta = data["weight_jsd_mean"] - data["count_jsd_mean"]
        lines.append(f"  {table:20s} {data['total_queries']:>8d} {data['n_cols']:>5d} "
                    f"{data['count_jsd_mean']:>10.4f} {data['weight_jsd_mean']:>11.4f} "
                    f"{delta:>+8.4f}")

    # Composite ranking: pruning-weighted JSD × log(volume)
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  TOP TABLES: Pruning-Weighted JSD × log(volume)")
    lines.append(f"{'=' * 90}")

    # Use best of weekly/daily
    best = {}
    for table, data in weekly_results.items():
        best[table] = {"jsd": data["weight_jsd_mean"], "gran": "Weekly", **data}
    for table, data in daily_results.items():
        if table not in best or data["weight_jsd_mean"] > best[table]["jsd"]:
            best[table] = {"jsd": data["weight_jsd_mean"], "gran": "Daily", **data}

    final = []
    for table, data in best.items():
        score = data["jsd"] * np.log(data["total_queries"] + 1)
        qpd = data["total_queries"] / 91
        # Also get mean selectivity for this table
        tstat = stats[stats["table"] == table]
        mean_sel = tstat["mean_sel"].mean() if not tstat.empty else float("nan")
        final.append((score, table, data, qpd, mean_sel))
    final.sort(reverse=True)

    for i, (score, table, data, qpd, mean_sel) in enumerate(final[:10]):
        lines.append(f"  #{i+1:2d} {table:20s} score={score:.3f}  "
                    f"queries={data['total_queries']:>6d} ({qpd:.1f}/day)  "
                    f"cols={data['n_cols']}  wJSD={data['jsd']:.4f}  "
                    f"mean_sel={mean_sel:.3f}  gran={data['gran']}")

    # Generate plots for top 6 tables
    plot_tables = [t for _, t, _, _, _ in final[:6]]
    plot_selectivity_timelines(fdf, plot_tables, cluster_label, out_dir)

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imdb", "tpch", "all", "provisioned"], default="all")
    parser.add_argument("--strategy", choices=["generation", "matching"], default="generation")
    args = parser.parse_args()

    out_dir = "work/output/selectivity_analysis"
    os.makedirs(out_dir, exist_ok=True)

    imdb_clusters = [
        ("work/output_c104_d0", 104, 0, "IMDB 104/0"),
        ("work/output_c85_d0", 85, 0, "IMDB 85/0"),
        ("work/output_c134_d0", 134, 0, "IMDB 134/0"),
        ("work/output_c55_d0", 55, 0, "IMDB 55/0"),
        ("work/output_select_full", 0, 0, "IMDB 0/0"),
        ("work/output_c19_d0", 19, 0, "IMDB 19/0"),
        ("work/output_c128_d0", 128, 0, "IMDB 128/0"),
        ("work/output_c105_d0", 105, 0, "IMDB 105/0"),
        ("work/output_c126_d0", 126, 0, "IMDB 126/0"),
        ("work/output_c129_d0", 129, 0, "IMDB 129/0"),
    ]

    tpch_clusters = [
        ("work/output_tpch_sf1_c134_d0", 134, 0, "TPC-H 134/0"),
        ("work/output_tpch_sf1_c104_d0", 104, 0, "TPC-H 104/0"),
        ("work/output_tpch_sf1_c55_d0", 55, 0, "TPC-H 55/0"),
        ("work/output_tpch_sf1_c85_d0", 85, 0, "TPC-H 85/0"),
        ("work/output_tpch_sf1_c19_d0", 19, 0, "TPC-H 19/0"),
        ("work/output_tpch_sf1_c128_d0", 128, 0, "TPC-H 128/0"),
        ("work/output_tpch_sf1_c105_d0", 105, 0, "TPC-H 105/0"),
        ("work/output_tpch_sf1_c0_d0", 0, 0, "TPC-H 0/0"),
        ("work/output_tpch_sf1_c126_d0", 126, 0, "TPC-H 126/0"),
    ]

    provisioned_clusters = [
        ("work/output_prov_c158_d0", 158, 0, "Prov-IMDB 158/0"),
        ("work/output_prov_c4_d0", 4, 0, "Prov-IMDB 4/0"),
        ("work/output_prov_c49_d0", 49, 0, "Prov-IMDB 49/0"),
        ("work/output_prov_c109_d0", 109, 0, "Prov-IMDB 109/0"),
        ("work/output_prov_c100_d0", 100, 0, "Prov-IMDB 100/0"),
    ]

    clusters = []
    if args.dataset in ("imdb", "all"):
        clusters.extend(imdb_clusters)
    if args.dataset in ("tpch", "all"):
        clusters.extend(tpch_clusters)
    if args.dataset in ("provisioned", "all"):
        clusters.extend(provisioned_clusters)

    all_text = []
    for out, cid, did, label in clusters:
        result = analyze_cluster(out, cid, did, label, out_dir, strategy=args.strategy)
        if result:
            all_text.append(result)

    report_path = os.path.join(out_dir, "selectivity_analysis.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(all_text))
    print(f"\nWritten: {report_path}")


if __name__ == "__main__":
    main()
