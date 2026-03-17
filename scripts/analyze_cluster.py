"""
Analyze a cluster's generated + matched workloads for filter diversity and temporal patterns.
Usage: python analyze_cluster.py <output_dir> <cluster_id> <db_id>
Writes results to <output_dir>/filter_analysis/
"""
import sys, os, ast, json
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_selectivities(sel_str):
    """Parse approximated_scan_selectivities column."""
    try:
        return ast.literal_eval(sel_str)
    except:
        return []

def analyze_workload(wl, label, out_dir):
    """Analyze a single workload (matched or generated)."""
    results = []
    results.append(f"={'='*70}")
    results.append(f"  {label} WORKLOAD ANALYSIS")
    results.append(f"={'='*70}")
    results.append(f"Total SELECT rows: {len(wl)}")
    
    # SQL string distribution
    if "sql" in wl.columns:
        sql_col = "sql"
    elif "query" in wl.columns:
        sql_col = "query"
    else:
        sql_col = None
    
    if sql_col:
        sql_counts = wl[sql_col].value_counts()
        results.append(f"Distinct SQL strings: {len(sql_counts)}")
        results.append(f"Most frequent: {sql_counts.iloc[0]}x ({sql_counts.iloc[0]/len(wl)*100:.1f}%)")
        if len(sql_counts) > 1:
            results.append(f"2nd most frequent: {sql_counts.iloc[1]}x ({sql_counts.iloc[1]/len(wl)*100:.1f}%)")
        once = (sql_counts == 1).sum()
        results.append(f"Appearing exactly once: {once}")
        
        # Coverage
        cumsum = sql_counts.cumsum()
        for pct in [50, 80, 90]:
            n = (cumsum <= len(wl) * pct / 100).sum() + 1
            results.append(f"  {pct}% coverage: top {n} SQL strings")
    
    # Parse selectivities
    if "approximated_scan_selectivities" in wl.columns:
        all_sels = []
        table_col_sels = defaultdict(list)  # (table, col) -> [selectivities]
        table_queries = defaultdict(int)
        table_col_queries = defaultdict(int)
        
        for _, row in wl.iterrows():
            sels = parse_selectivities(row["approximated_scan_selectivities"])
            for table, col, sel in sels:
                # Strip _0, _1 suffixes for cleaner table names
                base_table = table.rsplit("_", 1)[0] if table[-1].isdigit() and "_" in table else table
                all_sels.append(sel)
                table_col_sels[(base_table, col)].append(sel)
                table_queries[base_table] += 1
                table_col_queries[(base_table, col)] += 1
        
        results.append(f"\nTotal filter predicates: {len(all_sels)}")
        results.append(f"Distinct (table, col) pairs: {len(table_col_sels)}")
        results.append(f"Distinct tables with filters: {len(table_queries)}")
        
        if all_sels:
            all_sels_arr = np.array(all_sels)
            results.append(f"\nOverall selectivity distribution:")
            results.append(f"  min={all_sels_arr.min():.4f} p5={np.percentile(all_sels_arr,5):.4f} "
                         f"p25={np.percentile(all_sels_arr,25):.4f} median={np.median(all_sels_arr):.4f} "
                         f"p75={np.percentile(all_sels_arr,75):.4f} p95={np.percentile(all_sels_arr,95):.4f} "
                         f"max={all_sels_arr.max():.4f}")
        
        # Per-table analysis
        results.append(f"\n{'='*70}")
        results.append(f"  PER-TABLE FILTER ANALYSIS")
        results.append(f"{'='*70}")
        
        # Group by base table
        table_data = {}
        for (table, col), sels in table_col_sels.items():
            if table not in table_data:
                table_data[table] = {"cols": {}, "total_queries": table_queries[table]}
            table_data[table]["cols"][col] = sels
        
        # Sort by total queries descending
        for table in sorted(table_data, key=lambda t: table_data[t]["total_queries"], reverse=True):
            td = table_data[table]
            n_cols = len(td["cols"])
            results.append(f"\n  {table} ({td['total_queries']} queries, {n_cols} filter columns):")
            
            # Per-column selectivity stats
            selective_cols = 0
            for col in sorted(td["cols"]):
                sels_arr = np.array(td["cols"][col])
                med = np.median(sels_arr)
                is_selective = med < 0.5
                if is_selective:
                    selective_cols += 1
                marker = "✓" if is_selective else "✗"
                results.append(f"    {marker} {col:30s}: n={len(sels_arr):4d}  median={med:.3f}  "
                             f"min={sels_arr.min():.3f}  max={sels_arr.max():.3f}  "
                             f"p25={np.percentile(sels_arr,25):.3f}  p75={np.percentile(sels_arr,75):.3f}")
            
            results.append(f"    -> {selective_cols}/{n_cols} columns with median selectivity < 0.5")
    
    # Temporal analysis
    if "arrival_timestamp" in wl.columns and "approximated_scan_selectivities" in wl.columns:
        results.append(f"\n{'='*70}")
        results.append(f"  TEMPORAL FILTER CHANGES (weekly)")
        results.append(f"{'='*70}")
        
        wl_copy = wl.copy()
        wl_copy["ts"] = pd.to_datetime(wl_copy["arrival_timestamp"], format="ISO8601")
        wl_copy["week"] = wl_copy["ts"].dt.isocalendar().week.astype(int)
        
        # Per-table temporal analysis
        table_weekly = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for _, row in wl_copy.iterrows():
            week = row["week"]
            sels = parse_selectivities(row["approximated_scan_selectivities"])
            for table, col, sel in sels:
                base_table = table.rsplit("_", 1)[0] if table[-1].isdigit() and "_" in table else table
                table_weekly[base_table][week][col] += 1
        
        weeks = sorted(set(w for tw in table_weekly.values() for w in tw))
        n_weeks = len(weeks)
        
        # Summary table
        results.append(f"\n{'table':25s} {'queries':>8s} {'cols':>5s} {'sel_cols':>9s} {'q/day':>6s} {'wk_changes':>11s}")
        results.append("-" * 70)
        
        table_summaries = []
        for table in sorted(table_weekly, key=lambda t: table_queries.get(t, 0), reverse=True):
            tw = table_weekly[table]
            all_cols_ever = set()
            weekly_col_sets = []
            for w in weeks:
                cols_this_week = set(tw[w].keys()) if w in tw else set()
                all_cols_ever |= cols_this_week
                weekly_col_sets.append(cols_this_week)
            
            changes = sum(1 for i in range(1, len(weekly_col_sets)) 
                         if weekly_col_sets[i] != weekly_col_sets[i-1] 
                         and weekly_col_sets[i] and weekly_col_sets[i-1])
            
            # Count selective columns
            sel_cols = 0
            for col in all_cols_ever:
                if (table, col) in table_col_sels:
                    med = np.median(table_col_sels[(table, col)])
                    if med < 0.5:
                        sel_cols += 1
            
            tq = table_queries.get(table, 0)
            days = 91  # ~3 months
            table_summaries.append((table, tq, len(all_cols_ever), sel_cols, tq/days, changes, n_weeks-1))
            results.append(f"  {table:23s} {tq:>8d} {len(all_cols_ever):>5d} {sel_cols:>9d} {tq/days:>6.1f} {changes:>5d}/{n_weeks-1}")
        
        # Identify most promising tables
        results.append(f"\n{'='*70}")
        results.append(f"  MOST PROMISING TABLES FOR DYNAMIC CLUSTERING")
        results.append(f"{'='*70}")
        
        # Score: selective_cols * changes * log(queries+1)
        scored = []
        for table, tq, n_cols, sel_cols, qpd, changes, max_changes in table_summaries:
            if sel_cols >= 2 and changes >= 1:
                score = sel_cols * changes * np.log(tq + 1)
                scored.append((score, table, tq, n_cols, sel_cols, qpd, changes, max_changes))
        
        scored.sort(reverse=True)
        for i, (score, table, tq, n_cols, sel_cols, qpd, changes, max_changes) in enumerate(scored[:10]):
            results.append(f"  #{i+1} {table}: score={score:.1f} "
                         f"(queries={tq}, cols={n_cols}, selective={sel_cols}, "
                         f"q/day={qpd:.1f}, changes={changes}/{max_changes})")
    
    return "\n".join(results)


def main():
    if len(sys.argv) < 4:
        print("Usage: python analyze_cluster.py <output_dir> <cluster_id> <db_id>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    cluster_id = int(sys.argv[2])
    db_id = int(sys.argv[3])
    
    analysis_dir = os.path.join(output_dir, "filter_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    results = []
    results.append(f"CLUSTER {cluster_id} / DATABASE {db_id} ANALYSIS")
    results.append(f"Output dir: {output_dir}")
    results.append("")
    
    # Find generated workload
    gen_base = os.path.join(output_dir, "generated_workloads", "imdb", "serverless",
                           f"cluster_{cluster_id}", f"database_{db_id}")
    if os.path.exists(gen_base):
        gen_dirs = [d for d in os.listdir(gen_base) if d.startswith("generation_")]
        if gen_dirs:
            gen_wl_path = os.path.join(gen_base, gen_dirs[0], "workload.csv")
            if os.path.exists(gen_wl_path):
                gen_wl = pd.read_csv(gen_wl_path)
                results.append(analyze_workload(gen_wl, "GENERATED", analysis_dir))
                results.append("")
    
    # Find matched workload
    match_base = os.path.join(output_dir, "matched_workloads", "imdb", "serverless",
                             f"cluster_{cluster_id}", f"database_{db_id}")
    if not os.path.exists(match_base):
        # Try alternate path structure
        match_base = os.path.join(output_dir, "generated_workloads", "imdb", "serverless",
                                 f"cluster_{cluster_id}", f"database_{db_id}")
    
    # Search for matching workload files
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f == "workload.csv" and "matching_" in root:
                match_wl_path = os.path.join(root, f)
                match_wl = pd.read_csv(match_wl_path)
                results.append(analyze_workload(match_wl, "MATCHED", analysis_dir))
                results.append("")
                break
    
    output_file = os.path.join(analysis_dir, f"cluster_{cluster_id}_db_{db_id}_analysis.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    
    print(f"Analysis written to {output_file}")
    print(f"Generated workload found: {'yes' if 'gen_wl' in dir() else 'check paths'}")


if __name__ == "__main__":
    main()
