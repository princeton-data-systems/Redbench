#!/usr/bin/env python3
"""
Screen Redset clusters for QIG longevity.

For each cluster with sufficient usable SELECTs, compute:
- How many QIGs span the full 3-month period
- How many QIGs are active on >50% of days
- What fraction of total query volume comes from long-lived QIGs
- Whether the cluster has periodic QIG patterns (same QIGs recurring weekly)

Goal: find clusters where the workload is dominated by recurring, long-lived
query patterns (dashboarding workloads) rather than short-lived bursts.
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict


def qhash(row):
    """Build QIG hash from Redset columns."""
    scanset = tuple(sorted(map(int, row['read_table_ids'].split(',')))) if pd.notna(row['read_table_ids']) else ()
    return f"{row['feature_fingerprint']}#{row['num_scans']}#{row['num_joins']}#{scanset}"


def analyze_cluster(cdf, instance_id, dataset_type):
    """Analyze QIG longevity for one cluster."""
    cdf = cdf.copy()
    cdf['ts'] = pd.to_datetime(cdf['arrival_timestamp'])
    cdf['day'] = cdf['ts'].dt.date
    cdf['hour_of_week'] = cdf['ts'].dt.weekday * 24 + cdf['ts'].dt.hour

    total_days = (cdf['ts'].max() - cdf['ts'].min()).total_seconds() / 86400
    if total_days < 30:
        return None

    # Build QIG hash
    cdf['qhash'] = cdf.apply(qhash, axis=1)

    n_qigs = cdf['qhash'].nunique()
    n_queries = len(cdf)

    # QIG stats
    qig_stats = cdf.groupby('qhash').agg(
        count=('ts', 'size'),
        first=('ts', 'min'),
        last=('ts', 'max'),
        n_days=('day', 'nunique'),
    ).reset_index()
    qig_stats['lifespan_days'] = (qig_stats['last'] - qig_stats['first']).dt.total_seconds() / 86400
    qig_stats['daily_coverage'] = qig_stats['n_days'] / total_days

    # Longevity buckets
    full_span = qig_stats['lifespan_days'] > 80
    half_span = qig_stats['lifespan_days'] > 45
    week_plus = qig_stats['lifespan_days'] > 7
    short_lived = qig_stats['lifespan_days'] <= 1

    # Volume from long-lived QIGs
    vol_full = qig_stats.loc[full_span, 'count'].sum()
    vol_half = qig_stats.loc[half_span, 'count'].sum()
    vol_week = qig_stats.loc[week_plus, 'count'].sum()

    # QIGs active on >50% of days (truly recurring)
    recurring = qig_stats['daily_coverage'] > 0.5
    vol_recurring = qig_stats.loc[recurring, 'count'].sum()
    n_recurring = recurring.sum()

    # Weekly periodicity check: for long-lived QIGs, do they appear at
    # consistent hours-of-week?
    long_lived_qigs = qig_stats[half_span]['qhash'].values
    periodic_qig_count = 0
    if len(long_lived_qigs) > 0:
        for qh in long_lived_qigs[:50]:  # check top 50 to keep it fast
            qdf = cdf[cdf['qhash'] == qh]
            # Check if queries cluster at specific hours-of-week
            how_counts = qdf['hour_of_week'].value_counts()
            n_weeks = max(1, total_days / 7)
            # A QIG is "periodic" if it has at least one hour-of-week where
            # it appears in >50% of weeks
            if (how_counts > n_weeks * 0.5).any():
                periodic_qig_count += 1

    return {
        'instance_id': instance_id,
        'dataset': dataset_type,
        'n_queries': n_queries,
        'n_qigs': n_qigs,
        'total_days': total_days,
        'n_full_span': full_span.sum(),
        'n_half_span': half_span.sum(),
        'n_week_plus': week_plus.sum(),
        'n_short_lived': short_lived.sum(),
        'vol_full_pct': 100 * vol_full / n_queries,
        'vol_half_pct': 100 * vol_half / n_queries,
        'vol_week_pct': 100 * vol_week / n_queries,
        'n_recurring': n_recurring,
        'vol_recurring_pct': 100 * vol_recurring / n_queries,
        'n_periodic': periodic_qig_count,
        'n_periodic_checked': min(50, len(long_lived_qigs)),
    }


def main():
    min_usable_selects = 1000

    results = []

    for dataset_type, path in [
        ('serverless', 'work/data/full_serverless.parquet'),
        ('provisioned', 'work/data/full_provisioned.parquet'),
    ]:
        print(f"\nLoading {path}...")
        df = pd.read_parquet(path, columns=[
            'instance_id', 'database_id', 'feature_fingerprint',
            'query_type', 'num_joins', 'num_scans', 'read_table_ids',
            'arrival_timestamp', 'mbytes_scanned',
        ])

        # Filter to usable SELECTs
        sel = df[
            (df['query_type'] == 'select') &
            df['read_table_ids'].notna() &
            df['mbytes_scanned'].notna()
        ].copy()
        print(f"  {len(sel)} usable SELECTs across {sel['instance_id'].nunique()} clusters")

        # Group by cluster, filter by minimum volume
        cluster_counts = sel.groupby('instance_id').size()
        viable = cluster_counts[cluster_counts >= min_usable_selects].index
        print(f"  {len(viable)} clusters with >= {min_usable_selects} usable SELECTs")

        for i, cid in enumerate(sorted(viable)):
            cdf = sel[sel['instance_id'] == cid]
            # Use database 0 (primary) if it has enough data, otherwise all
            db0 = cdf[cdf['database_id'] == 0]
            if len(db0) >= min_usable_selects:
                cdf = db0
            # For very large clusters, sample to keep runtime reasonable
            # but sample uniformly across time to preserve temporal patterns
            max_rows = 500000
            if len(cdf) > max_rows:
                cdf = cdf.sample(n=max_rows, random_state=42)
            print(f"  [{i+1}/{len(viable)}] Cluster {cid} ({dataset_type}): {len(cdf)} queries...", end="", flush=True)
            result = analyze_cluster(cdf, cid, dataset_type)
            if result:
                results.append(result)
                print(f" {result['n_half_span']} QIGs >45d, {result['vol_half_pct']:.1f}% vol, "
                      f"{result['n_recurring']} recurring")
            else:
                print(" skipped (too short)")

    # Sort by volume from long-lived QIGs
    results.sort(key=lambda x: x['vol_recurring_pct'], reverse=True)

    # Print summary
    print(f"\n{'='*140}")
    print("QIG LONGEVITY SCREENING — ALL CLUSTERS")
    print(f"{'='*140}")
    print(f"{'type':>6s} {'cid':>5s} {'queries':>8s} {'QIGs':>6s} {'days':>5s} "
          f"{'full':>5s} {'half':>5s} {'>7d':>5s} {'<1d':>5s} "
          f"{'%vol_full':>9s} {'%vol_half':>9s} {'%vol_>7d':>9s} "
          f"{'recur':>6s} {'%vol_rec':>8s} {'periodic':>8s}")
    print("-" * 140)

    for r in results:
        print(f"{r['dataset']:>6s} {r['instance_id']:>5d} {r['n_queries']:>8d} {r['n_qigs']:>6d} "
              f"{r['total_days']:>5.0f} "
              f"{r['n_full_span']:>5d} {r['n_half_span']:>5d} {r['n_week_plus']:>5d} {r['n_short_lived']:>5d} "
              f"{r['vol_full_pct']:>8.1f}% {r['vol_half_pct']:>8.1f}% {r['vol_week_pct']:>8.1f}% "
              f"{r['n_recurring']:>6d} {r['vol_recurring_pct']:>7.1f}% "
              f"{r['n_periodic']:>4d}/{r['n_periodic_checked']:<3d}")

    # Save CSV
    out_path = "work/output/qig_longevity_screening.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
