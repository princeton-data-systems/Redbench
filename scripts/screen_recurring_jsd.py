#!/usr/bin/env python3
"""
For clusters with high recurring QIG volume, check whether the recurring QIGs
produce column-level diversity (different QIGs -> different scansets) and
whether their relative proportions shift over time (JSD).

This answers: can we get high JSD *and* high periodicity from the same cluster?
"""

import pandas as pd
import numpy as np
import math
from collections import defaultdict


def qhash(row):
    scanset = tuple(sorted(map(int, row['read_table_ids'].split(',')))) if pd.notna(row['read_table_ids']) else ()
    return f"{row['feature_fingerprint']}#{row['num_scans']}#{row['num_joins']}#{scanset}"


def jsd(p, q):
    """Jensen-Shannon distance between two dicts."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in all_keys}
    kl_pm = sum(p.get(k, 0) * math.log(p[k] / m[k]) for k in all_keys if p.get(k, 0) > 0 and m[k] > 0)
    kl_qm = sum(q.get(k, 0) * math.log(q[k] / m[k]) for k in all_keys if q.get(k, 0) > 0 and m[k] > 0)
    return math.sqrt(max(0, (kl_pm + kl_qm) / 2))


def analyze_cluster(cdf, instance_id, dataset_type):
    cdf = cdf.copy()
    cdf['ts'] = pd.to_datetime(cdf['arrival_timestamp'])
    cdf['day'] = cdf['ts'].dt.date
    cdf['qhash'] = cdf.apply(qhash, axis=1)

    total_days = (cdf['ts'].max() - cdf['ts'].min()).total_seconds() / 86400
    if total_days < 30:
        return None

    # Identify recurring QIGs (active >50% of days)
    qig_stats = cdf.groupby('qhash').agg(
        count=('ts', 'size'),
        n_days=('day', 'nunique'),
    ).reset_index()
    qig_stats['daily_coverage'] = qig_stats['n_days'] / total_days
    recurring_qigs = set(qig_stats[qig_stats['daily_coverage'] > 0.5]['qhash'])

    if len(recurring_qigs) < 2:
        return None

    recurring_df = cdf[cdf['qhash'].isin(recurring_qigs)]
    recurring_vol = len(recurring_df)
    recurring_pct = 100 * recurring_vol / len(cdf)

    # For each recurring QIG, what scanset does it access?
    # (scanset = the set of table IDs it reads)
    qig_scansets = {}
    for qh in recurring_qigs:
        sample = cdf[cdf['qhash'] == qh].iloc[0]
        scanset = sample['read_table_ids']
        qig_scansets[qh] = scanset

    # How many distinct scansets among recurring QIGs?
    distinct_scansets = len(set(qig_scansets.values()))

    # Daily scanset distribution: for each day, what fraction of recurring
    # queries go to each scanset?
    recurring_df = recurring_df.copy()
    recurring_df['scanset'] = recurring_df['qhash'].map(qig_scansets)

    days = sorted(recurring_df['day'].unique())
    if len(days) < 7:
        return None

    # Compute daily scanset distributions and JSD between consecutive days
    daily_dists = []
    for d in days:
        ddf = recurring_df[recurring_df['day'] == d]
        if len(ddf) < 5:
            continue
        counts = ddf['scanset'].value_counts()
        total = counts.sum()
        dist = {k: v / total for k, v in counts.items()}
        daily_dists.append((d, dist, len(ddf)))

    if len(daily_dists) < 7:
        return None

    jsds = []
    for i in range(1, len(daily_dists)):
        j = jsd(daily_dists[i-1][1], daily_dists[i][1])
        jsds.append(j)

    mean_jsd = np.mean(jsds) if jsds else 0
    median_jsd = np.median(jsds) if jsds else 0
    max_jsd = np.max(jsds) if jsds else 0

    # Also compute weekly JSD (aggregate by week)
    recurring_df_copy = recurring_df.copy()
    min_date = recurring_df_copy['ts'].min()
    recurring_df_copy['week'] = ((recurring_df_copy['ts'] - min_date).dt.days // 7) + 1
    weeks = sorted(recurring_df_copy['week'].unique())

    weekly_dists = []
    for w in weeks:
        wdf = recurring_df_copy[recurring_df_copy['week'] == w]
        if len(wdf) < 10:
            continue
        counts = wdf['scanset'].value_counts()
        total = counts.sum()
        dist = {k: v / total for k, v in counts.items()}
        weekly_dists.append((w, dist, len(wdf)))

    weekly_jsds = []
    for i in range(1, len(weekly_dists)):
        j = jsd(weekly_dists[i-1][1], weekly_dists[i][1])
        weekly_jsds.append(j)

    weekly_mean_jsd = np.mean(weekly_jsds) if weekly_jsds else 0

    # Per-scanset volume breakdown
    scanset_counts = recurring_df['scanset'].value_counts()
    top_scansets = scanset_counts.head(5)

    return {
        'instance_id': instance_id,
        'dataset': dataset_type,
        'n_queries': len(cdf),
        'n_recurring_qigs': len(recurring_qigs),
        'recurring_vol_pct': recurring_pct,
        'distinct_scansets': distinct_scansets,
        'daily_jsd_mean': mean_jsd,
        'daily_jsd_median': median_jsd,
        'daily_jsd_max': max_jsd,
        'weekly_jsd_mean': weekly_mean_jsd,
        'recurring_qpd': recurring_vol / total_days,
        'top_scansets': top_scansets.to_dict(),
        'n_top_scansets': len(top_scansets),
    }


def main():
    # Clusters identified as having high recurring QIG volume
    target_clusters = {
        'serverless': [85, 131, 104, 126, 130],
        'provisioned': [74, 4, 96, 185, 29, 113, 0, 1, 27, 28, 109, 99, 55, 8, 77, 178, 186, 161, 144, 51, 7, 66, 31],
    }

    results = []

    for dataset_type, path in [
        ('serverless', 'work/data/full_serverless.parquet'),
        ('provisioned', 'work/data/full_provisioned.parquet'),
    ]:
        if not target_clusters.get(dataset_type):
            continue

        print(f"\nLoading {path}...")
        df = pd.read_parquet(path, columns=[
            'instance_id', 'database_id', 'feature_fingerprint',
            'query_type', 'num_joins', 'num_scans', 'read_table_ids',
            'arrival_timestamp', 'mbytes_scanned',
        ])

        sel = df[
            (df['query_type'] == 'select') &
            df['read_table_ids'].notna() &
            df['mbytes_scanned'].notna()
        ].copy()

        for cid in target_clusters[dataset_type]:
            cdf = sel[sel['instance_id'] == cid]
            db0 = cdf[cdf['database_id'] == 0]
            if len(db0) >= 500:
                cdf = db0

            # Sample large clusters
            if len(cdf) > 500000:
                cdf = cdf.sample(n=500000, random_state=42)

            print(f"  Cluster {cid} ({dataset_type}): {len(cdf)} queries...", end="", flush=True)
            result = analyze_cluster(cdf, cid, dataset_type)
            if result:
                results.append(result)
                print(f" {result['n_recurring_qigs']} recurring QIGs, "
                      f"{result['distinct_scansets']} scansets, "
                      f"daily JSD={result['daily_jsd_mean']:.3f}, "
                      f"weekly JSD={result['weekly_jsd_mean']:.3f}")
            else:
                print(" skipped (insufficient recurring QIGs or data)")

    # Sort by daily JSD
    results.sort(key=lambda x: x['daily_jsd_mean'], reverse=True)

    print(f"\n{'='*130}")
    print("RECURRING QIG JSD ANALYSIS — Can we get high JSD + high periodicity?")
    print(f"{'='*130}")
    print(f"{'type':>6s} {'cid':>5s} {'queries':>8s} {'rec_QIGs':>8s} {'%vol_rec':>8s} "
          f"{'scansets':>8s} {'d_JSD_mean':>10s} {'d_JSD_med':>10s} {'d_JSD_max':>10s} "
          f"{'w_JSD_mean':>10s} {'rec_q/day':>9s}")
    print("-" * 130)

    for r in results:
        print(f"{r['dataset']:>6s} {r['instance_id']:>5d} {r['n_queries']:>8d} "
              f"{r['n_recurring_qigs']:>8d} {r['recurring_vol_pct']:>7.1f}% "
              f"{r['distinct_scansets']:>8d} {r['daily_jsd_mean']:>10.4f} "
              f"{r['daily_jsd_median']:>10.4f} {r['daily_jsd_max']:>10.4f} "
              f"{r['weekly_jsd_mean']:>10.4f} {r['rec_q/day']:>9.1f}")

    # Save
    out_path = "work/output/recurring_jsd_screening.csv"
    out_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'top_scansets'} for r in results])
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
