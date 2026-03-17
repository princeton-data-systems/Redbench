"""
Analyze proportional shifts in filter column usage across all clusters.
For each table, measure how much the *distribution* of filter columns changes
over time (weekly, daily, 6-hourly), even if the *set* of columns stays the same.

Uses Jensen-Shannon divergence to quantify distributional shift between
consecutive time windows.
"""
import sys, os, ast, glob, json
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_sels(s):
    try:
        return ast.literal_eval(s)
    except:
        return []


def find_gen_workload(output_dir, cluster_id, db_id, db_name=None):
    if db_name:
        db_names = [db_name]
    else:
        db_names = ["imdb", "tpch"]
    for name in db_names:
        for redset_type in ["serverless", "provisioned"]:
            pattern = os.path.join(output_dir, "generated_workloads", name, redset_type,
                                   f"cluster_{cluster_id}", f"database_{db_id}",
                                   "generation_*", "workload.csv")
            paths = glob.glob(pattern)
            if paths:
                return paths[0]
    return None


def compute_distribution_shifts(wl, time_col, granularity_label):
    """
    For each table, compute the filter-column distribution in each time window,
    then measure JSD between consecutive windows.
    Returns: {table: {n_cols, total_queries, mean_jsd, max_jsd, jsd_series, window_distributions}}
    """
    results = {}

    # Parse all selectivities and build per-row table->col mapping
    rows_data = []
    for _, row in wl.iterrows():
        sels = parse_sels(row.get("approximated_scan_selectivities", "[]"))
        tw = row[time_col]
        for table, col, sel in sels:
            base = table.rsplit("_", 1)[0] if table[-1].isdigit() and "_" in table else table
            rows_data.append((base, col, tw))

    if not rows_data:
        return results

    df = pd.DataFrame(rows_data, columns=["table", "col", "window"])

    for table, tdf in df.groupby("table"):
        # Get all columns ever seen
        all_cols = sorted(tdf["col"].unique())
        if len(all_cols) < 2:
            continue  # Need at least 2 columns for meaningful distribution

        windows = sorted(tdf["window"].unique())
        if len(windows) < 2:
            continue

        # Build distribution per window
        distributions = []
        window_labels = []
        for w in windows:
            wdf = tdf[tdf["window"] == w]
            counts = wdf["col"].value_counts()
            dist = np.array([counts.get(c, 0) for c in all_cols], dtype=float)
            total = dist.sum()
            if total > 0:
                dist = dist / total
                distributions.append(dist)
                window_labels.append(w)

        if len(distributions) < 2:
            continue

        # Compute JSD between consecutive windows
        jsds = []
        for i in range(1, len(distributions)):
            jsd = jensenshannon(distributions[i-1], distributions[i])
            if not np.isnan(jsd):
                jsds.append(jsd)

        if not jsds:
            continue

        results[table] = {
            "n_cols": len(all_cols),
            "columns": all_cols,
            "total_queries": len(tdf),
            "mean_jsd": np.mean(jsds),
            "max_jsd": np.max(jsds),
            "median_jsd": np.median(jsds),
            "p75_jsd": np.percentile(jsds, 75),
            "jsd_series": jsds,
            "window_labels": window_labels,
            "distributions": distributions,
            "n_windows": len(distributions),
        }

    return results


def analyze_cluster(wl_path, cluster_label, out_dir):
    """Analyze a single cluster's generated workload."""
    wl = pd.read_csv(wl_path)
    wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")
    wl["week"] = wl["ts"].dt.isocalendar().week.astype(int)
    wl["day"] = wl["ts"].dt.date
    wl["hour_6"] = wl["ts"].dt.floor("6h")

    results_text = []
    results_text.append(f"\n{'#'*80}")
    results_text.append(f"  {cluster_label} — {len(wl)} generated queries")
    results_text.append(f"{'#'*80}")

    # Analyze at multiple granularities
    granularities = [
        ("week", "Weekly"),
        ("day", "Daily"),
        ("hour_6", "6-Hourly"),
    ]

    cluster_tables = {}  # table -> best granularity results

    for time_col, gran_label in granularities:
        shifts = compute_distribution_shifts(wl, time_col, gran_label)

        results_text.append(f"\n{'='*70}")
        results_text.append(f"  {gran_label} Proportional Shifts (Jensen-Shannon Divergence)")
        results_text.append(f"{'='*70}")
        results_text.append(f"  {'table':25s} {'queries':>8s} {'cols':>5s} {'windows':>8s} "
                          f"{'mean_JSD':>9s} {'med_JSD':>8s} {'p75_JSD':>8s} {'max_JSD':>8s}")
        results_text.append(f"  {'-'*25} {'-'*8} {'-'*5} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

        # Sort by mean_jsd * log(queries) to balance shift magnitude with volume
        scored = []
        for table, data in shifts.items():
            score = data["mean_jsd"] * np.log(data["total_queries"] + 1)
            scored.append((score, table, data))
        scored.sort(reverse=True)

        for score, table, data in scored:
            results_text.append(
                f"  {table:25s} {data['total_queries']:>8d} {data['n_cols']:>5d} "
                f"{data['n_windows']:>8d} {data['mean_jsd']:>9.4f} {data['median_jsd']:>8.4f} "
                f"{data['p75_jsd']:>8.4f} {data['max_jsd']:>8.4f}"
            )

            # Track best result per table
            if table not in cluster_tables or data["mean_jsd"] > cluster_tables[table].get("best_mean_jsd", 0):
                cluster_tables[table] = {
                    "best_mean_jsd": data["mean_jsd"],
                    "best_granularity": gran_label,
                    **data,
                }

    # Identify most promising tables: high volume + high JSD
    results_text.append(f"\n{'='*70}")
    results_text.append(f"  TOP TABLES FOR DYNAMIC CLUSTERING (volume × shift)")
    results_text.append(f"{'='*70}")

    final_scored = []
    for table, data in cluster_tables.items():
        # Score: mean_jsd * log(total_queries) — rewards both shift and volume
        score = data["best_mean_jsd"] * np.log(data["total_queries"] + 1)
        final_scored.append((score, table, data))
    final_scored.sort(reverse=True)

    for i, (score, table, data) in enumerate(final_scored[:15]):
        qpd = data["total_queries"] / 91
        results_text.append(
            f"  #{i+1:2d} {table:25s} score={score:.3f}  "
            f"queries={data['total_queries']:>6d} ({qpd:.1f}/day)  "
            f"cols={data['n_cols']}  mean_JSD={data['best_mean_jsd']:.4f}  "
            f"gran={data['best_granularity']}"
        )

    # Generate per-table stacked area plots for top tables
    plot_tables = [t for _, t, d in final_scored[:6]]
    if plot_tables:
        _plot_proportional_timelines(wl, plot_tables, cluster_label, out_dir)

    return "\n".join(results_text), final_scored


def _plot_proportional_timelines(wl, tables, cluster_label, out_dir):
    """Create stacked area plots showing filter column proportions over time."""
    # Parse all selectivities
    rows_data = []
    for _, row in wl.iterrows():
        sels = parse_sels(row.get("approximated_scan_selectivities", "[]"))
        ts = row["ts"]
        for table, col, sel in sels:
            base = table.rsplit("_", 1)[0] if table[-1].isdigit() and "_" in table else table
            if base in tables:
                rows_data.append((base, col, ts))

    if not rows_data:
        return

    df = pd.DataFrame(rows_data, columns=["table", "col", "ts"])
    # Use relative week numbers (1-based) from the start of the data
    min_date = df["ts"].min()
    df["week"] = ((df["ts"] - min_date).dt.days // 7) + 1

    for table in tables:
        tdf = df[df["table"] == table]
        if len(tdf) < 10:
            continue

        all_cols = sorted(tdf["col"].unique())
        weeks = sorted(tdf["week"].unique())

        # Build proportions matrix
        props = np.zeros((len(weeks), len(all_cols)))
        counts_per_week = np.zeros(len(weeks))
        for i, w in enumerate(weeks):
            wdf = tdf[tdf["week"] == w]
            counts_per_week[i] = len(wdf)
            for j, c in enumerate(all_cols):
                props[i, j] = (wdf["col"] == c).sum()
            row_total = props[i].sum()
            if row_total > 0:
                props[i] /= row_total

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1])
        fig.suptitle(f"{cluster_label} — {table} filter column proportions (weekly)",
                     fontsize=13, fontweight="bold")

        # Stacked area
        ax1.stackplot(weeks, props.T, labels=all_cols, alpha=0.8)
        ax1.set_ylabel("Proportion")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper left", fontsize=7, ncol=min(5, len(all_cols)))
        ax1.set_xlim(weeks[0], weeks[-1])

        # Volume bar chart
        ax2.bar(weeks, counts_per_week, color="steelblue", alpha=0.7)
        ax2.set_ylabel("Queries")
        ax2.set_xlabel("Week")
        ax2.set_xlim(weeks[0] - 0.5, weeks[-1] + 0.5)

        plt.tight_layout()
        safe_label = cluster_label.lower().replace(" ", "_").replace("/", "_")
        fname = os.path.join(out_dir, f"{safe_label}_{table}_proportions.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imdb", "tpch", "all", "provisioned"], default="all",
                        help="Which dataset to analyze")
    args = parser.parse_args()

    out_dir = "work/output/proportional_analysis"
    os.makedirs(out_dir, exist_ok=True)

    imdb_clusters = [
        ("work/output_select_full", 0, 0, "IMDB 0/0"),
        ("work/output_c85_d0", 85, 0, "IMDB 85/0"),
        ("work/output_c104_d0", 104, 0, "IMDB 104/0"),
        ("work/output_c134_d0", 134, 0, "IMDB 134/0"),
        ("work/output_c55_d0", 55, 0, "IMDB 55/0"),
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

    all_results = []
    all_scored = {}

    for output_dir, cid, did, label in clusters:
        wl_path = find_gen_workload(output_dir, cid, did)
        if wl_path is None:
            print(f"  WARNING: No workload found for {label}")
            continue

        print(f"Analyzing {label} from {wl_path}...")
        text, scored = analyze_cluster(wl_path, label, out_dir)
        all_results.append(text)
        all_scored[label] = scored
        print(f"  Done — {len(scored)} tables with distributional shifts")

    # Write combined results
    output_file = os.path.join(out_dir, "proportional_shift_analysis.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(all_results))
    print(f"\nResults written to {output_file}")

    # Cross-cluster comparison: for each table, show which cluster gives best shift
    print("\n" + "="*80)
    print("  CROSS-CLUSTER COMPARISON: Best cluster per table")
    print("="*80)

    table_best = defaultdict(list)
    for label, scored in all_scored.items():
        for score, table, data in scored:
            table_best[table].append((score, label, data))

    # Sort tables by best score across any cluster
    table_max_score = {t: max(s for s, _, _ in entries) for t, entries in table_best.items()}
    for table in sorted(table_max_score, key=table_max_score.get, reverse=True)[:20]:
        entries = sorted(table_best[table], reverse=True)
        best_score, best_cluster, best_data = entries[0]
        qpd = best_data["total_queries"] / 91
        print(f"  {table:25s} best={best_cluster:15s} score={best_score:.3f} "
              f"queries={best_data['total_queries']:>6d} ({qpd:.1f}/day) "
              f"cols={best_data['n_cols']} mean_JSD={best_data['best_mean_jsd']:.4f}")


if __name__ == "__main__":
    main()
