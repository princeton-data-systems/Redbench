#!/usr/bin/env python3
"""
Periodic Workload Analysis for Sort Key Optimization.

For each table in each cluster's generated workload:

1. Build a vector time series at 6-hour granularity where component i is
   the total skipping volume for column i: sum(1 - selectivity).
   6-hour bins avoid sparsity issues from hourly data while still capturing
   intra-day patterns.

2. Split the time series in half: training period and test period.

3. Fit three models on the training period:
   - Model 1 (Static Average): predict the overall mean vector for every bin.
   - Model 2 (Periodic Floor): fold at candidate periods, take a low quantile
     per phase bin, smooth, and predict the floor template.
   - Model 3 (Periodic Average): fold at candidate periods, take the mean
     per phase bin, and predict the full average template.

4. Evaluate each model on the test period using median relative L1 error.

5. Classify the table:
   - Category 1 (Completely Nonstationary): static average wins.
   - Category 2 (Stationary Periodic, Nonstationary Aperiodic): floor wins.
   - Category 3 (Completely Stationary): periodic average wins.

6. Report the classification, the dominant period, and the fraction of
   total skipping volume captured by the periodic component (Y%).

Time granularity: 6-hour bins.
  - Daily period  = 4 bins per cycle
  - Weekly period = 28 bins per cycle
"""

import ast
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Parameters ──────────────────────────────────────────────────────────
BIN_HOURS = 6                          # aggregate into 6-hour bins
# Candidate periods in bins (not hours)
CANDIDATE_PERIODS_BINS = {
    "daily": 24 // BIN_HOURS,          # 4 bins
    "2-day": 48 // BIN_HOURS,          # 8 bins
    "weekly": 168 // BIN_HOURS,        # 28 bins
    "2-week": 336 // BIN_HOURS,        # 56 bins
}
PERIOD_HOURS = {"daily": 24, "2-day": 48, "weekly": 168, "2-week": 336}
FLOOR_QUANTILE = 0.25                  # quantile for periodic floor
FLOOR_QUANTILE_MANY_CYCLES = 0.10      # use if > 90 cycles in training
FLOOR_SMOOTH_WINDOW = 3                # moving-average window for floor smoothing
SIMILARITY_TOLERANCE = 0.03            # models within this → prefer simpler
MIN_BINS = 48 // BIN_HOURS             # minimum bins for a table (8 = 2 days)
MIN_OBS_PER_BIN = 5                    # minimum filter observations per bin


# ── Data loading helpers ────────────────────────────────────────────────

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


def load_filter_observations(wl_path):
    """Load workload CSV → DataFrame with (table, col, selectivity, ts, hour, pruning)."""
    wl = pd.read_csv(wl_path, low_memory=False)
    wl = wl[wl["query_type"] == "select"].copy()
    wl["ts"] = pd.to_datetime(wl["arrival_timestamp"], format="ISO8601")

    sel_col = wl["approximated_scan_selectivities"].fillna("[]")
    ts_col = wl["ts"]

    tables, cols, sels, timestamps = [], [], [], []
    for sel_str, ts in zip(sel_col.values, ts_col.values):
        for table, col, sel in parse_sels(sel_str):
            tables.append(table)
            cols.append(col)
            sels.append(sel)
            timestamps.append(ts)

    fdf = pd.DataFrame({"table": tables, "col": cols, "selectivity": sels, "ts": timestamps})
    fdf["ts"] = pd.to_datetime(fdf["ts"])
    fdf["hour"] = fdf["ts"].dt.floor("h")
    fdf["pruning"] = 1.0 - fdf["selectivity"]
    return fdf



# ── Build binned vector time series ─────────────────────────────────────

def build_vector_timeseries(tdf, all_cols):
    """
    Build a complete vector time series at BIN_HOURS granularity.

    Returns (matrix, bin_index, obs_counts) where:
      - matrix: shape (n_bins, n_cols), skipping volume per (bin, col)
      - bin_index: DatetimeIndex of bin start times
      - obs_counts: shape (n_bins,), filter observations per bin
    Or (None, None, None) if insufficient data.
    """
    n_cols = len(all_cols)
    col_idx = {c: i for i, c in enumerate(all_cols)}

    # Build hourly matrix first, then aggregate into bins
    hour_min = tdf["hour"].min()
    hour_max = tdf["hour"].max()
    full_hours = pd.date_range(hour_min, hour_max, freq="h")
    n_hours = len(full_hours)

    hourly_matrix = np.zeros((n_hours, n_cols))
    hourly_obs = np.zeros(n_hours)
    hour_to_idx = {h: i for i, h in enumerate(full_hours)}

    hourly_col = tdf.groupby(["hour", "col"])["pruning"].sum().reset_index()
    for _, row in hourly_col.iterrows():
        h_idx = hour_to_idx.get(row["hour"])
        c_idx = col_idx.get(row["col"])
        if h_idx is not None and c_idx is not None:
            hourly_matrix[h_idx, c_idx] = row["pruning"]

    hourly_obs_series = tdf.groupby("hour").size()
    for h, count in hourly_obs_series.items():
        h_idx = hour_to_idx.get(h)
        if h_idx is not None:
            hourly_obs[h_idx] = count

    # Aggregate into BIN_HOURS bins
    n_bins = n_hours // BIN_HOURS
    if n_bins < MIN_BINS:
        return None, None, None

    trimmed_hours = n_bins * BIN_HOURS
    matrix = hourly_matrix[:trimmed_hours].reshape(n_bins, BIN_HOURS, n_cols).sum(axis=1)
    obs_counts = hourly_obs[:trimmed_hours].reshape(n_bins, BIN_HOURS).sum(axis=1)

    bin_index = full_hours[::BIN_HOURS][:n_bins]

    n_valid = (obs_counts >= MIN_OBS_PER_BIN).sum()
    if n_valid < MIN_BINS:
        return None, None, None

    return matrix, bin_index, obs_counts


# ── Model 1: Static Average ────────────────────────────────────────────

def fit_static_average(train_matrix):
    """Compute the overall mean vector across all training bins."""
    return train_matrix.mean(axis=0)


def predict_static(mean_vec, n_bins):
    """Predict the same mean vector for every bin."""
    return np.tile(mean_vec, (n_bins, 1))


# ── Model 2: Periodic Floor ────────────────────────────────────────────

def fit_periodic_floor(train_matrix, period_bins):
    """
    Fold training data at the given period (in bins), compute a low quantile
    per phase bin (component-wise), and smooth.
    Returns floor_template: shape (period_bins, n_cols).
    """
    n_bins, n_cols = train_matrix.shape
    n_cycles = n_bins / period_bins
    q = FLOOR_QUANTILE_MANY_CYCLES if n_cycles > 90 else FLOOR_QUANTILE

    phase_groups = defaultdict(list)
    for t in range(n_bins):
        phase_groups[t % period_bins].append(t)

    floor_template = np.zeros((period_bins, n_cols))
    for h in range(period_bins):
        if phase_groups[h]:
            vectors = train_matrix[phase_groups[h]]
            floor_template[h] = np.quantile(vectors, q, axis=0)

    # Circular moving-average smoothing
    w = FLOOR_SMOOTH_WINDOW
    smoothed = np.zeros_like(floor_template)
    for h in range(period_bins):
        indices = [(h + d) % period_bins for d in range(-w // 2, w // 2 + 1)]
        smoothed[h] = floor_template[indices].mean(axis=0)

    return np.maximum(smoothed, 0.0)


# ── Model 3: Periodic Average ──────────────────────────────────────────

def fit_periodic_average(train_matrix, period_bins):
    """
    Fold training data at the given period, compute the mean per phase bin.
    Returns avg_template: shape (period_bins, n_cols).
    """
    n_bins = train_matrix.shape[0]
    phase_groups = defaultdict(list)
    for t in range(n_bins):
        phase_groups[t % period_bins].append(t)

    avg_template = np.zeros((period_bins, train_matrix.shape[1]))
    for h in range(period_bins):
        if phase_groups[h]:
            avg_template[h] = train_matrix[phase_groups[h]].mean(axis=0)

    return avg_template


def fit_periodic_average_multi(train_matrix, period_bins_list):
    """
    Multi-period additive model: overall_mean + sum of deviations per period.
    Returns (overall_mean, list of deviation_templates).
    """
    overall_mean = train_matrix.mean(axis=0)
    n_bins = train_matrix.shape[0]

    deviation_templates = []
    for period_bins in period_bins_list:
        phase_groups = defaultdict(list)
        for t in range(n_bins):
            phase_groups[t % period_bins].append(t)

        template = np.zeros((period_bins, train_matrix.shape[1]))
        for h in range(period_bins):
            if phase_groups[h]:
                template[h] = train_matrix[phase_groups[h]].mean(axis=0) - overall_mean
        deviation_templates.append(template)

    return overall_mean, deviation_templates


# ── Prediction helpers ──────────────────────────────────────────────────

def predict_periodic(template, period_bins, n_bins):
    """Tile a periodic template to cover n_bins."""
    reps = (n_bins // period_bins) + 2
    return np.tile(template, (reps, 1))[:n_bins]


def predict_multi(overall_mean, dev_templates, period_bins_list, n_bins):
    """Predict using additive multi-period model."""
    pred = np.tile(overall_mean, (n_bins, 1))
    for template, period_bins in zip(dev_templates, period_bins_list):
        for t in range(n_bins):
            pred[t] += template[t % period_bins]
    return pred



# ── Evaluation ──────────────────────────────────────────────────────────

def relative_l1_error(observed, predicted):
    """
    Per-bin relative L1 error: |v(t) - v_hat(t)|_1 / |v(t)|_1.
    Returns array of per-bin errors (NaN where observed is all-zero).
    """
    abs_diff = np.abs(observed - predicted).sum(axis=1)
    obs_norm = np.abs(observed).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = np.where(obs_norm > 0, abs_diff / obs_norm, np.nan)
    return errors


def median_relative_error(observed, predicted):
    """Median of per-bin relative L1 errors, ignoring NaN."""
    errors = relative_l1_error(observed, predicted)
    valid = errors[np.isfinite(errors)]
    return np.median(valid) if len(valid) > 0 else np.inf


# ── Classification ──────────────────────────────────────────────────────

def classify_table(errors_by_model):
    """
    Classify based on which model achieves lowest median error.
    Prefer simpler models when performance is similar (within SIMILARITY_TOLERANCE).
    """
    static_err = errors_by_model.get("static", np.inf)

    floor_items = [(k, v) for k, v in errors_by_model.items() if k.startswith("floor_")]
    avg_items = [(k, v) for k, v in errors_by_model.items() if k.startswith("avg_")]

    best_floor = min(floor_items, key=lambda x: x[1]) if floor_items else (None, np.inf)
    best_avg = min(avg_items, key=lambda x: x[1]) if avg_items else (None, np.inf)

    overall_best_err = min(static_err, best_floor[1], best_avg[1])

    # Prefer simpler: static < floor < avg
    if static_err <= overall_best_err + SIMILARITY_TOLERANCE:
        return 1, "static", static_err
    if best_floor[1] <= overall_best_err + SIMILARITY_TOLERANCE:
        return 2, best_floor[0], best_floor[1]
    return 3, best_avg[0], best_avg[1]


# ── Y% metric ──────────────────────────────────────────────────────────

def compute_y_pct(template, period_bins, test_matrix):
    """
    Y% = sum of predicted skipping volume / sum of observed skipping volume.
    Prediction is clipped to not exceed observed (floor can't capture more
    than what actually happened).
    """
    n_bins = test_matrix.shape[0]
    pred = predict_periodic(template, period_bins, n_bins)
    pred_clipped = np.minimum(pred, test_matrix)
    total_obs = test_matrix.sum()
    return pred_clipped.sum() / total_obs if total_obs > 0 else 0.0


# ── Sort key regret metric ──────────────────────────────────────────────
# Cost model: at each bin, the sort key is one column. The benefit is the
# skipping volume for that column in that bin (data pruning is only possible
# if the sort key matches the filtered column). Switching the sort key
# between consecutive bins costs SWITCH_COST.
#
# Given a sequence of skipping volume vectors, the optimal sort key schedule
# is found by dynamic programming. Regret compares the benefit achieved by
# a model-based policy to the oracle (which has perfect knowledge).

SWITCH_COST = 50.0  # cost of changing the sort key between bins


def optimal_sort_key_schedule(matrix, switch_cost=SWITCH_COST):
    """
    DP to find the optimal sort key column at each bin.
    matrix: shape (n_bins, n_cols), skipping volume per (bin, col).
    Returns (total_benefit, schedule) where schedule[t] is the chosen column index.
    """
    n_bins, n_cols = matrix.shape
    if n_bins == 0 or n_cols == 0:
        return 0.0, []

    # dp[t][c] = best total benefit achievable from bin 0..t if sort key at bin t is c
    dp = np.full((n_bins, n_cols), -np.inf)
    dp[0, :] = matrix[0, :]  # benefit at bin 0 for each column choice

    parent = np.zeros((n_bins, n_cols), dtype=int)

    for t in range(1, n_bins):
        for c in range(n_cols):
            benefit = matrix[t, c]
            # Option 1: stay on same column as previous bin (no switch cost)
            # Option 2: switch from some other column (pay switch_cost)
            best_prev = -np.inf
            best_prev_col = c
            for prev_c in range(n_cols):
                cost = 0.0 if prev_c == c else switch_cost
                val = dp[t - 1, prev_c] - cost
                if val > best_prev:
                    best_prev = val
                    best_prev_col = prev_c
            dp[t, c] = best_prev + benefit
            parent[t, c] = best_prev_col

    # Trace back
    best_final_col = np.argmax(dp[n_bins - 1])
    total_benefit = dp[n_bins - 1, best_final_col]

    schedule = [0] * n_bins
    schedule[n_bins - 1] = best_final_col
    for t in range(n_bins - 2, -1, -1):
        schedule[t] = parent[t + 1, schedule[t + 1]]

    return float(total_benefit), schedule


def sort_key_benefit(matrix, schedule):
    """Compute total benefit of a sort key schedule (without switch costs)."""
    benefit = 0.0
    for t, c in enumerate(schedule):
        benefit += matrix[t, c]
    return benefit


def sort_key_regret(test_matrix, predicted_matrix, switch_cost=SWITCH_COST):
    """
    Compute regret: how much worse is the model-based sort key schedule
    compared to the oracle?

    1. Oracle: run DP on test_matrix (perfect knowledge).
    2. Policy: run DP on predicted_matrix, then evaluate that schedule on test_matrix.
    3. Regret = 1 - (policy_benefit / oracle_benefit).

    Returns (regret, oracle_benefit, policy_benefit, oracle_schedule, policy_schedule).
    """
    oracle_benefit, oracle_schedule = optimal_sort_key_schedule(test_matrix, switch_cost)
    _, policy_schedule = optimal_sort_key_schedule(predicted_matrix, switch_cost)

    # Evaluate policy schedule on actual data (including switch costs)
    n_bins = test_matrix.shape[0]
    policy_benefit = 0.0
    for t in range(n_bins):
        policy_benefit += test_matrix[t, policy_schedule[t]]
        if t > 0 and policy_schedule[t] != policy_schedule[t - 1]:
            policy_benefit -= switch_cost

    if oracle_benefit <= 0:
        return 0.0, oracle_benefit, policy_benefit, oracle_schedule, policy_schedule

    regret = 1.0 - (policy_benefit / oracle_benefit)
    return regret, oracle_benefit, policy_benefit, oracle_schedule, policy_schedule



# ── Per-table analysis ──────────────────────────────────────────────────

def classify_segment(train_seg, test_seg):
    """
    Fit all models on train_seg, evaluate on test_seg, return category.
    Returns (category, best_model_name) or (None, None) if insufficient data.
    """
    n_cols = train_seg.shape[1]
    if train_seg.shape[0] < MIN_BINS or test_seg.shape[0] < MIN_BINS:
        return None, None

    errors = {}
    # Static
    static_mean = fit_static_average(train_seg)
    errors["static"] = median_relative_error(
        test_seg, predict_static(static_mean, test_seg.shape[0]))

    # Floor and average for each period
    for name, period_bins in CANDIDATE_PERIODS_BINS.items():
        if train_seg.shape[0] < 2 * period_bins:
            continue
        tmpl = fit_periodic_floor(train_seg, period_bins)
        errors[f"floor_{name}"] = median_relative_error(
            test_seg, predict_periodic(tmpl, period_bins, test_seg.shape[0]))
        tmpl = fit_periodic_average(train_seg, period_bins)
        errors[f"avg_{name}"] = median_relative_error(
            test_seg, predict_periodic(tmpl, period_bins, test_seg.shape[0]))

    # Multi-period average
    viable = [(n, p) for n, p in CANDIDATE_PERIODS_BINS.items()
              if train_seg.shape[0] >= 2 * p]
    if len(viable) >= 2:
        pbl = [p for _, p in viable]
        om, dt = fit_periodic_average_multi(train_seg, pbl)
        errors["avg_multi"] = median_relative_error(
            test_seg, predict_multi(om, dt, pbl, test_seg.shape[0]))

    if not errors:
        return None, None
    cat, model, _ = classify_table(errors)
    return cat, model


def three_split_stability(matrix):
    """
    Split data into thirds. Classify on seg1→seg2 and seg2→seg3.
    Returns (cat_12, cat_23, stable_cat) where stable_cat is the category
    if both agree, else None.
    """
    n = matrix.shape[0]
    s1 = n // 3
    s2 = 2 * n // 3
    seg1 = matrix[:s1]
    seg2 = matrix[s1:s2]
    seg3 = matrix[s2:]

    cat_12, _ = classify_segment(seg1, seg2)
    cat_23, _ = classify_segment(seg2, seg3)

    stable = cat_12 if (cat_12 is not None and cat_12 == cat_23) else None
    return cat_12, cat_23, stable

def analyze_table(tdf):
    """
    Full analysis for one table: build models, evaluate, classify.
    Returns result dict or None if insufficient data.
    """
    all_cols = sorted(tdf["col"].unique())
    if len(all_cols) < 2:
        return None

    matrix, bin_index, obs_counts = build_vector_timeseries(tdf, all_cols)
    if matrix is None:
        return None

    n_bins, n_cols = matrix.shape

    # Split in half
    split = n_bins // 2
    train = matrix[:split]
    test = matrix[split:]

    if train.shape[0] < MIN_BINS or test.shape[0] < MIN_BINS:
        return None

    # ── Model 1: Static Average ─────────────────────────────────────────
    static_mean = fit_static_average(train)
    static_pred_test = predict_static(static_mean, test.shape[0])
    static_pred_train = predict_static(static_mean, train.shape[0])
    predictions = {"static": static_pred_test}
    models = {"static": {"mean": static_mean}}

    # ── Model 2: Periodic Floor (each candidate period) ─────────────────
    best_floor_name = None
    best_floor_template = None
    best_floor_period_bins = None

    for name, period_bins in CANDIDATE_PERIODS_BINS.items():
        if train.shape[0] < 2 * period_bins:
            continue
        template = fit_periodic_floor(train, period_bins)
        pred_test = predict_periodic(template, period_bins, test.shape[0])
        key = f"floor_{name}"
        predictions[key] = pred_test
        models[key] = {"template": template, "period_bins": period_bins, "name": name}
        if best_floor_name is None:
            best_floor_name = name
            best_floor_template = template
            best_floor_period_bins = period_bins

    # ── Model 3: Periodic Average (each candidate period) ───────────────
    best_avg_name = None
    best_avg_template = None
    best_avg_period_bins = None

    for name, period_bins in CANDIDATE_PERIODS_BINS.items():
        if train.shape[0] < 2 * period_bins:
            continue
        template = fit_periodic_average(train, period_bins)
        pred_test = predict_periodic(template, period_bins, test.shape[0])
        key = f"avg_{name}"
        predictions[key] = pred_test
        models[key] = {"template": template, "period_bins": period_bins, "name": name}
        if best_avg_name is None:
            best_avg_name = name
            best_avg_template = template
            best_avg_period_bins = period_bins

    # Multi-period average
    viable = [(n, p) for n, p in CANDIDATE_PERIODS_BINS.items()
              if train.shape[0] >= 2 * p]
    if len(viable) >= 2:
        period_bins_list = [p for _, p in viable]
        overall_mean, dev_templates = fit_periodic_average_multi(train, period_bins_list)
        multi_pred_test = predict_multi(overall_mean, dev_templates, period_bins_list,
                                         test.shape[0])
        key = "avg_multi"
        predictions[key] = multi_pred_test
        models[key] = {"overall_mean": overall_mean, "dev_templates": dev_templates,
                        "period_bins_list": period_bins_list}

    # ── Compute train-set predictions (for model selection) ─────────────
    train_predictions = {"static": static_pred_train}
    for key, m in models.items():
        if key == "static":
            continue
        if "template" in m:
            train_predictions[key] = predict_periodic(
                m["template"], m["period_bins"], train.shape[0])
        elif "overall_mean" in m:
            train_predictions[key] = predict_multi(
                m["overall_mean"], m["dev_templates"],
                m["period_bins_list"], train.shape[0])

    # ── L1 errors on train (for regret model selection) and test (for L1 classification + reporting)
    errors_train = {}
    errors_test = {}
    for model_name in predictions:
        errors_train[model_name] = median_relative_error(train, train_predictions[model_name])
        errors_test[model_name] = median_relative_error(test, predictions[model_name])

    # ── Classify by L1 error (using test-set errors) ────────────────────
    # Forward direction: train on first half, evaluate on second half.
    category_l1, best_model_l1, best_err_l1 = classify_table(errors_test)

    # ── Reverse direction: train on second half, evaluate on first half ─
    # This enables a symmetry check: "true" periodic tables are those
    # classified the same way regardless of which half is used for fitting.
    rev_models_errors = {"static": median_relative_error(
        train, predict_static(fit_static_average(test), train.shape[0]))}
    for name, period_bins in CANDIDATE_PERIODS_BINS.items():
        if test.shape[0] < 2 * period_bins:
            continue
        tmpl = fit_periodic_floor(test, period_bins)
        rev_models_errors[f"floor_{name}"] = median_relative_error(
            train, predict_periodic(tmpl, period_bins, train.shape[0]))
        tmpl = fit_periodic_average(test, period_bins)
        rev_models_errors[f"avg_{name}"] = median_relative_error(
            train, predict_periodic(tmpl, period_bins, train.shape[0]))
    viable_rev = [(n, p) for n, p in CANDIDATE_PERIODS_BINS.items()
                  if test.shape[0] >= 2 * p]
    if len(viable_rev) >= 2:
        pbl = [p for _, p in viable_rev]
        om, dt = fit_periodic_average_multi(test, pbl)
        rev_models_errors["avg_multi"] = median_relative_error(
            train, predict_multi(om, dt, pbl, train.shape[0]))
    category_rev, _, _ = classify_table(rev_models_errors)

    # True category: same in both directions
    true_cat = category_l1 if category_l1 == category_rev else None

    # Update best_floor/best_avg based on test-set L1
    best_floor_err_test = np.inf
    for name in CANDIDATE_PERIODS_BINS:
        key = f"floor_{name}"
        if key in errors_test and errors_test[key] < best_floor_err_test:
            best_floor_err_test = errors_test[key]
            best_floor_name = name
            best_floor_template = models[key]["template"]
            best_floor_period_bins = models[key]["period_bins"]
    best_avg_err_test = np.inf
    for name in CANDIDATE_PERIODS_BINS:
        key = f"avg_{name}"
        if key in errors_test and errors_test[key] < best_avg_err_test:
            best_avg_err_test = errors_test[key]
            best_avg_name = name
            best_avg_template = models[key]["template"]
            best_avg_period_bins = models[key]["period_bins"]
    if "avg_multi" in errors_test and errors_test["avg_multi"] < best_avg_err_test:
        best_avg_name = "multi"

    # ── Classification margin (L1, on test) ─────────────────────────────
    static_err_test = errors_test.get("static", np.inf)
    best_floor_err_t = min(
        (v for k, v in errors_test.items() if k.startswith("floor_")), default=np.inf)
    best_avg_err_t = min(
        (v for k, v in errors_test.items() if k.startswith("avg_")), default=np.inf)
    category_errors = {1: static_err_test, 2: best_floor_err_t, 3: best_avg_err_t}
    sorted_cats = sorted(category_errors.items(), key=lambda x: x[1])
    margin_l1 = sorted_cats[1][1] - sorted_cats[0][1] if len(sorted_cats) >= 2 else 0.0

    # ── Sort key regret for each model ─────────────────────────────────
    # Train regret: DP on train predictions, evaluate on train.
    #   Used for model selection (regret-based classification).
    # Test regret: DP on test predictions, evaluate on test.
    #   Used for reporting generalization performance.
    # Oracle: DP directly on actual test data (perfect foresight), regret = 0.

    # Train-set regret (for model selection)
    regrets_train = {}
    for model_name, pred in train_predictions.items():
        regret, _, _, _, _ = sort_key_regret(train, pred)
        regrets_train[model_name] = regret

    # Test-set regret (for reporting)
    regrets_test = {}
    for model_name, pred in predictions.items():
        regret, _, _, _, _ = sort_key_regret(test, pred)
        regrets_test[model_name] = regret
    oracle_benefit = sort_key_regret(test, test)[1]

    # ── Classify by regret (using train-set regret for selection) ───────
    category_regret, best_model_regret, _ = classify_table(regrets_train)
    # Report the test-set regret of the train-selected model
    best_regret = regrets_test.get(best_model_regret, 0)

    # Regret-based classification margin (on train set)
    static_regret = regrets_train.get("static", np.inf)
    best_floor_regret = min(
        (v for k, v in regrets_train.items() if k.startswith("floor_")), default=np.inf)
    best_avg_regret = min(
        (v for k, v in regrets_train.items() if k.startswith("avg_")), default=np.inf)
    cat_regrets = {1: static_regret, 2: best_floor_regret, 3: best_avg_regret}
    sorted_cat_regrets = sorted(cat_regrets.items(), key=lambda x: x[1])
    margin_regret = (sorted_cat_regrets[1][1] - sorted_cat_regrets[0][1]
                     if len(sorted_cat_regrets) >= 2 else 0.0)

    # ── Y% for the L1-winning periodic model ────────────────────────────
    y_pct = 0.0
    winning_period_name = None
    if category_l1 == 2 and best_floor_template is not None:
        y_pct = compute_y_pct(best_floor_template, best_floor_period_bins, test)
        winning_period_name = best_floor_name
    elif category_l1 == 3 and best_avg_template is not None:
        y_pct = compute_y_pct(best_avg_template, best_avg_period_bins, test)
        winning_period_name = best_avg_name

    # ── Three-split stability check ────────────────────────────────────
    cat_12, cat_23, stable_cat = three_split_stability(matrix)

    # ── Build prediction for the L1-winning model (for plotting) ────────
    winning_pred = predictions.get(best_model_l1, static_pred_test)
    error_series = relative_l1_error(test, winning_pred)

    return {
        "columns": all_cols,
        "n_cols": n_cols,
        "n_bins": n_bins,
        "bin_hours": BIN_HOURS,
        "split": split,
        # L1-based classification (selected on train, reported on test)
        "category": category_l1,
        "category_rev": category_rev,
        "true_cat": true_cat,
        "best_model": best_model_l1,
        "best_error": best_err_l1,
        "all_errors": errors_test,
        "all_errors_train": errors_train,
        "margin": margin_l1,
        # Regret-based classification
        "category_regret": category_regret,
        "best_model_regret": best_model_regret,
        "best_regret": best_regret,
        "regrets": regrets_test,
        "regrets_train": regrets_train,
        "margin_regret": margin_regret,
        "oracle_benefit": oracle_benefit,
        # Other metrics
        "y_pct": y_pct,
        "winning_period_name": winning_period_name,
        "total_skipping_volume": float(matrix.sum()),
        # Three-split stability
        "cat_12": cat_12,
        "cat_23": cat_23,
        "stable_cat": stable_cat,
        # For plotting
        "matrix": matrix,
        "bin_index": bin_index,
        "obs_counts": obs_counts,
        "train": train,
        "test": test,
        "winning_pred": winning_pred,
        "error_series": error_series,
        "static_mean": static_mean,
        "best_floor_template": best_floor_template,
        "best_floor_period_bins": best_floor_period_bins,
        "best_floor_name": best_floor_name,
        "best_avg_template": best_avg_template,
        "best_avg_period_bins": best_avg_period_bins,
        "best_avg_name": best_avg_name,
    }



# ── Plotting ────────────────────────────────────────────────────────────

CATEGORY_NAMES = {
    1: "Cat 1: Completely Nonstationary",
    2: "Cat 2: Stationary Periodic",
    3: "Cat 3: Completely Stationary",
}
CATEGORY_COLORS = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}


def plot_table_analysis(table_name, result, cluster_label, out_dir):
    """
    Diagnostic plot for one table:
      1. Stacked area: column distribution over full period
      2. Total skipping volume + model prediction on test
      3. Per-bin relative error on test period
      4. Model comparison bar chart
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.5, 2, 1.5, 1.2],
                            width_ratios=[3, 1], hspace=0.35, wspace=0.3)

    cat = result["category"]
    cat_name = CATEGORY_NAMES[cat]
    period_name = result["winning_period_name"] or "—"

    fig.suptitle(
        f"{cluster_label} / {table_name}  —  {cat_name}\n"
        f"Best model: {result['best_model']} (median err={result['best_error']:.3f})  |  "
        f"Period: {period_name}  |  Y%={result['y_pct']:.1%}  |  "
        f"{result['n_cols']} cols, {result['n_bins']} bins ({BIN_HOURS}h each)",
        fontsize=11, y=0.995)

    all_cols = result["columns"]
    matrix = result["matrix"]
    n_bins = result["n_bins"]
    split = result["split"]
    bins = np.arange(n_bins)

    # ── Panel 1: stacked area of column proportions ─────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props = matrix / row_sums
    ax1.stackplot(bins, props.T, labels=all_cols, alpha=0.8)
    ax1.axvline(x=split, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.text(split - 2, 0.95, "← train", ha="right", fontsize=8)
    ax1.text(split + 2, 0.95, "test →", ha="left", fontsize=8)
    ax1.set_ylabel("Column proportion\n(skipping-weighted)")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, n_bins - 1)
    if len(all_cols) <= 12:
        ax1.legend(loc="upper left", fontsize=6, ncol=min(6, len(all_cols)))
    ax1.set_title("Column distribution over time", fontsize=9)

    # ── Panel 2: total skipping volume + prediction ─────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    total_vol = matrix.sum(axis=1)
    ax2.plot(bins, total_vol, linewidth=0.4, alpha=0.5, color="gray", label="observed")
    test_bins = bins[split:]
    pred_total = result["winning_pred"].sum(axis=1)
    ax2.plot(test_bins, pred_total, linewidth=1.0, color=CATEGORY_COLORS[cat],
             label=f"predicted ({result['best_model']})")
    ax2.axvline(x=split, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Total skipping\nvolume")
    ax2.set_xlim(0, n_bins - 1)
    ax2.legend(fontsize=7)
    ax2.set_title("Total skipping volume: observed vs predicted", fontsize=9)

    # ── Panel 2 right: model comparison ─────────────────────────────────
    ax2r = fig.add_subplot(gs[1, 1])
    model_names = list(result["all_errors"].keys())
    model_errors = [result["all_errors"][m] for m in model_names]
    colors = [CATEGORY_COLORS[cat] if m == result["best_model"] else "#bdc3c7"
              for m in model_names]
    ax2r.barh(range(len(model_names)), model_errors, color=colors,
              edgecolor="gray", linewidth=0.5)
    ax2r.set_yticks(range(len(model_names)))
    ax2r.set_yticklabels(model_names, fontsize=7)
    ax2r.set_xlabel("Median relative\nL1 error", fontsize=8)
    ax2r.invert_yaxis()
    ax2r.set_title("Model comparison", fontsize=9)

    # ── Panel 3: per-bin error on test ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    err = result["error_series"]
    valid = np.isfinite(err)
    ax3.plot(test_bins[valid], err[valid], linewidth=0.5, alpha=0.7,
             color=CATEGORY_COLORS[cat])
    if valid.sum() > 0:
        ax3.axhline(y=np.nanmedian(err), color="black", linestyle="--",
                     linewidth=0.8, alpha=0.5,
                     label=f"median={np.nanmedian(err):.3f}")
    ax3.set_ylabel("Relative L1\nerror")
    ax3.set_xlabel(f"Bin ({BIN_HOURS}h each)")
    ax3.set_xlim(split, n_bins - 1)
    ax3.legend(fontsize=7)
    ax3.set_title("Per-bin prediction error on test period", fontsize=9)

    # ── Panel 4: per-column skipping volume ─────────────────────────────
    ax4 = fig.add_subplot(gs[3, :])
    col_volumes = matrix.sum(axis=0)
    sorted_idx = np.argsort(col_volumes)[::-1]
    sorted_cols = [all_cols[i] for i in sorted_idx]
    sorted_vols = col_volumes[sorted_idx]
    ax4.barh(range(len(sorted_cols)), sorted_vols, color="steelblue", alpha=0.7)
    ax4.set_yticks(range(len(sorted_cols)))
    ax4.set_yticklabels(sorted_cols, fontsize=7)
    ax4.set_xlabel("Total skipping volume")
    ax4.invert_yaxis()
    ax4.set_title("Per-column total skipping volume", fontsize=9)

    plt.tight_layout()
    safe = cluster_label.lower().replace(" ", "_").replace("/", "_")
    fname = os.path.join(out_dir, f"{safe}_{table_name}_analysis.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname



# ── Cluster-level driver ────────────────────────────────────────────────

def analyze_cluster(output_dir, cluster_id, db_id, label, out_dir, plot=True):
    """Run analysis for one cluster. Returns dict of {table: result}."""
    wl_path, db_name = find_workload(output_dir, cluster_id, db_id)
    if not wl_path:
        print(f"SKIP {label}: workload not found")
        return None

    print(f"\nLoading {label} from {wl_path}...")
    fdf = load_filter_observations(wl_path)
    print(f"  {len(fdf)} filter observations, {fdf['table'].nunique()} tables")

    table_col_counts = fdf.groupby("table")["col"].nunique()
    multi_col_tables = table_col_counts[table_col_counts >= 2].index.tolist()

    cluster_results = {}
    for table in multi_col_tables:
        tdf = fdf[fdf["table"] == table]
        result = analyze_table(tdf)
        if result is None:
            continue
        cluster_results[table] = result
        cat_name = CATEGORY_NAMES[result["category"]]
        cat_name_r = CATEGORY_NAMES[result["category_regret"]]
        period = result["winning_period_name"] or "—"
        regret = result["regrets"].get(result["best_model"], 0)
        print(f"    {table:>20s}: L1→{cat_name}  Regret→{cat_name_r}  "
              f"err={result['best_error']:.3f}  regret={regret:.3f}  "
              f"period={period}  Y%={result['y_pct']:.1%}")

        if plot:
            plot_table_analysis(table, result, label, out_dir)

    return cluster_results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Periodic workload analysis for sort key optimization")
    parser.add_argument("--dataset", choices=["imdb", "tpch", "all", "provisioned"],
                        default="all")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    imdb_clusters = [
        ("work/output_c104_d0", 104, 0, "IMDB srvl-104"),
        ("work/output_c55_d0", 55, 0, "IMDB srvl-55"),
        ("work/output_c85_d0", 85, 0, "IMDB srvl-85"),
        ("work/output_c134_d0", 134, 0, "IMDB srvl-134"),
        ("work/output_c19_d0", 19, 0, "IMDB srvl-19"),
        ("work/output_c105_d0", 105, 0, "IMDB srvl-105"),
        ("work/output_c126_d0", 126, 0, "IMDB srvl-126"),
        ("work/output_c128_d0", 128, 0, "IMDB srvl-128"),
        ("work/output_c129_d0", 129, 0, "IMDB srvl-129"),
        ("work/output_select_full", 0, 0, "IMDB srvl-0"),
    ]
    tpch_clusters = [
        ("work/output_tpch_sf1_c104_d0", 104, 0, "TPC-H srvl-104"),
        ("work/output_tpch_sf1_c55_d0", 55, 0, "TPC-H srvl-55"),
        ("work/output_tpch_sf1_c85_d0", 85, 0, "TPC-H srvl-85"),
        ("work/output_tpch_sf1_c134_d0", 134, 0, "TPC-H srvl-134"),
        ("work/output_tpch_sf1_c19_d0", 19, 0, "TPC-H srvl-19"),
        ("work/output_tpch_sf1_c105_d0", 105, 0, "TPC-H srvl-105"),
        ("work/output_tpch_sf1_c126_d0", 126, 0, "TPC-H srvl-126"),
        ("work/output_tpch_sf1_c128_d0", 128, 0, "TPC-H srvl-128"),
        ("work/output_tpch_sf1_c0_d0", 0, 0, "TPC-H srvl-0"),
    ]
    provisioned_clusters = [
        ("work/output_prov_c158_d0", 158, 0, "IMDB prov-158"),
        ("work/output_prov_c4_d0", 4, 0, "IMDB prov-4"),
        ("work/output_prov_c49_d0", 49, 0, "IMDB prov-49"),
        ("work/output_prov_c109_d0", 109, 0, "IMDB prov-109"),
        ("work/output_prov_c100_d0", 100, 0, "IMDB prov-100"),
    ]

    clusters = []
    if args.dataset in ("imdb", "all"):
        clusters.extend(imdb_clusters)
    if args.dataset in ("tpch", "all"):
        clusters.extend(tpch_clusters)
    if args.dataset in ("provisioned", "all"):
        clusters.extend(provisioned_clusters)

    out_dir = "work/output/periodicity_decomposed"
    os.makedirs(out_dir, exist_ok=True)

    all_results = []  # (label, table, result)

    for output_dir, cluster_id, db_id, label in clusters:
        cluster_results = analyze_cluster(
            output_dir, cluster_id, db_id, label, out_dir, plot=not args.no_plot)
        if cluster_results:
            for table, result in cluster_results.items():
                all_results.append((label, table, result))

    # ── Summary report ──────────────────────────────────────────────────
    all_results.sort(key=lambda x: x[2]["best_error"])

    lines = []
    lines.append("=" * 160)
    lines.append("PERIODIC WORKLOAD ANALYSIS FOR SORT KEY OPTIMIZATION")
    lines.append(f"Bin size: {BIN_HOURS}h  |  Candidate periods: "
                 + ", ".join(f"{n} ({p} bins)" for n, p in CANDIDATE_PERIODS_BINS.items())
                 + f"  |  Floor quantile: {FLOOR_QUANTILE}"
                 + f"  |  Similarity tolerance: {SIMILARITY_TOLERANCE}")
    lines.append("=" * 160)

    lines.append(f"\n{'cluster':>20s} {'table':>15s} {'cat':>5s} {'best_model':>18s} "
                 f"{'med_err':>8s} {'static':>8s} "
                 + "  ".join(f"{'fl_'+n:>8s}" for n in CANDIDATE_PERIODS_BINS)
                 + "  "
                 + "  ".join(f"{'av_'+n:>8s}" for n in CANDIDATE_PERIODS_BINS)
                 + f"  {'av_multi':>8s} {'period':>8s} {'Y%':>6s} {'cols':>5s} {'skip_vol':>10s}")
    lines.append("-" * 200)

    for label, table, r in all_results:
        e = r["all_errors"]
        period = r["winning_period_name"] or "—"
        floor_cols = "  ".join(f"{e.get(f'floor_{n}', np.inf):>8.3f}" for n in CANDIDATE_PERIODS_BINS)
        avg_cols = "  ".join(f"{e.get(f'avg_{n}', np.inf):>8.3f}" for n in CANDIDATE_PERIODS_BINS)
        lines.append(
            f"{label:>20s} {table:>15s} {r['category']:>5d} "
            f"{r['best_model']:>18s} {r['best_error']:>8.3f} "
            f"{e.get('static', np.inf):>8.3f}  "
            f"{floor_cols}  {avg_cols}  "
            f"{e.get('avg_multi', np.inf):>8.3f} "
            f"{period:>8s} {r['y_pct']:>5.1%} {r['n_cols']:>5d} "
            f"{r['total_skipping_volume']:>10.0f}")

    # ── Category distribution (L1-based) ──────────────────────────────
    cat_counts = defaultdict(int)
    cat_volumes = defaultdict(float)
    for _, _, r in all_results:
        cat_counts[r["category"]] += 1
        cat_volumes[r["category"]] += r["total_skipping_volume"]

    total_tables = len(all_results)
    total_volume = sum(cat_volumes.values())

    lines.append(f"\n{'=' * 160}")
    lines.append("SUMMARY — L1-BASED CLASSIFICATION (model selected by test-set L1 — characterization, not deployment)")
    lines.append(f"{'=' * 160}")
    lines.append(f"  Tables analyzed: {total_tables}")
    for cat in [1, 2, 3]:
        n = cat_counts[cat]
        vol = cat_volumes[cat]
        pct_t = 100 * n / max(1, total_tables)
        pct_v = 100 * vol / max(1, total_volume)
        lines.append(f"  {CATEGORY_NAMES[cat]:>40s}: {n:>3d} tables ({pct_t:>5.1f}%)  "
                     f"skipping volume: {vol:>12.0f} ({pct_v:>5.1f}%)")

    # Period distribution for Cat 2+3
    period_counts = defaultdict(int)
    for _, _, r in all_results:
        if r["category"] in (2, 3) and r["winning_period_name"]:
            period_counts[r["winning_period_name"]] += 1
    if period_counts:
        lines.append(f"\n  Dominant periods (Cat 2 + Cat 3):")
        for p, c in sorted(period_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {p:>10s}: {c} tables")

    # Y% distribution
    y_pcts = [r["y_pct"] for _, _, r in all_results if r["category"] in (2, 3)]
    if y_pcts:
        lines.append(f"\n  Y% distribution (Cat 2 + Cat 3):")
        lines.append(f"    mean={np.mean(y_pcts):.1%}, median={np.median(y_pcts):.1%}, "
                     f"p25={np.percentile(y_pcts, 25):.1%}, "
                     f"p75={np.percentile(y_pcts, 75):.1%}")

    # L1 error distribution by category
    for cat in [1, 2, 3]:
        errs = [r["best_error"] for _, _, r in all_results if r["category"] == cat]
        if errs:
            lines.append(f"\n  Median L1 error for {CATEGORY_NAMES[cat]}:")
            lines.append(f"    mean={np.mean(errs):.3f}, median={np.median(errs):.3f}, "
                         f"min={np.min(errs):.3f}, max={np.max(errs):.3f}")

    # L1 classification margin
    lines.append(f"\n  L1 classification margin (error gap between best and runner-up category):")
    margins = [r["margin"] for _, _, r in all_results]
    lines.append(f"    mean={np.mean(margins):.3f}, median={np.median(margins):.3f}, "
                 f"p25={np.percentile(margins, 25):.3f}, p75={np.percentile(margins, 75):.3f}")
    narrow = sum(1 for m in margins if m < SIMILARITY_TOLERANCE)
    lines.append(f"    Narrow margin (< {SIMILARITY_TOLERANCE}): {narrow} / {len(margins)} tables")

    # ── Category distribution (regret-based) ────────────────────────────
    cat_counts_r = defaultdict(int)
    cat_volumes_r = defaultdict(float)
    for _, _, r in all_results:
        cat_counts_r[r["category_regret"]] += 1
        cat_volumes_r[r["category_regret"]] += r["total_skipping_volume"]

    lines.append(f"\n{'=' * 160}")
    lines.append("SUMMARY — REGRET-BASED CLASSIFICATION")
    lines.append(f"(Model selection by train-set regret. Reported regret is on test period. "
                 f"Switch cost = {SWITCH_COST}. Oracle = DP on actual test data, regret = 0.)")
    lines.append(f"{'=' * 160}")
    lines.append(f"  Tables analyzed: {total_tables}")
    for cat in [1, 2, 3]:
        n = cat_counts_r[cat]
        vol = cat_volumes_r[cat]
        pct_t = 100 * n / max(1, total_tables)
        pct_v = 100 * vol / max(1, total_volume)
        lines.append(f"  {CATEGORY_NAMES[cat]:>40s}: {n:>3d} tables ({pct_t:>5.1f}%)  "
                     f"skipping volume: {vol:>12.0f} ({pct_v:>5.1f}%)")

    # Regret by L1-category (how much regret does each L1-category have?)
    for cat in [1, 2, 3]:
        cat_regrets = [r["regrets"].get(r["best_model"], 0)
                       for _, _, r in all_results if r["category"] == cat]
        if cat_regrets:
            lines.append(f"\n  Test-set regret of L1-best model for {CATEGORY_NAMES[cat]}:")
            lines.append(f"    mean={np.mean(cat_regrets):.3f}, "
                         f"median={np.median(cat_regrets):.3f}, "
                         f"min={np.min(cat_regrets):.3f}, "
                         f"max={np.max(cat_regrets):.3f}")

    # Regret classification margin
    lines.append(f"\n  Regret classification margin:")
    margins_r = [r["margin_regret"] for _, _, r in all_results]
    lines.append(f"    mean={np.mean(margins_r):.3f}, median={np.median(margins_r):.3f}, "
                 f"p25={np.percentile(margins_r, 25):.3f}, "
                 f"p75={np.percentile(margins_r, 75):.3f}")
    narrow_r = sum(1 for m in margins_r if m < SIMILARITY_TOLERANCE)
    lines.append(f"    Narrow margin (< {SIMILARITY_TOLERANCE}): {narrow_r} / {len(margins_r)} tables")

    # Overall regret comparison
    lines.append(f"\n  Sort key regret comparison (all {total_tables} tables):")
    lines.append(f"    All models trained on first half. Regret evaluated on second half.")
    lines.append(f"    Oracle: DP on actual test data. Regret = 0 by definition.")
    lines.append(f"    Static: train mean → DP picks one column → evaluate on test.")
    lines.append(f"    Best-L1: model selected by L1 on test (characterization) → DP on its predictions → evaluate on test.")
    lines.append(f"    Best-regret: model selected by regret on train → DP on its test predictions → evaluate on test.")
    static_regrets = [r["regrets"].get("static", 0) for _, _, r in all_results]
    best_l1_regrets = [r["regrets"].get(r["best_model"], 0) for _, _, r in all_results]
    best_r_regrets = [r["best_regret"] for _, _, r in all_results]
    lines.append(f"    Static:      mean={np.mean(static_regrets):.3f}, "
                 f"median={np.median(static_regrets):.3f}")
    lines.append(f"    Best-L1:     mean={np.mean(best_l1_regrets):.3f}, "
                 f"median={np.median(best_l1_regrets):.3f}")
    lines.append(f"    Best-regret: mean={np.mean(best_r_regrets):.3f}, "
                 f"median={np.median(best_r_regrets):.3f}")

    # Agreement between L1 and regret classifications
    agree = sum(1 for _, _, r in all_results if r["category"] == r["category_regret"])
    lines.append(f"\n  L1 vs regret classification agreement: "
                 f"{agree} / {total_tables} tables ({100*agree/max(1,total_tables):.0f}%)")

    # ── Three-split stability ───────────────────────────────────────────
    lines.append(f"\n{'=' * 160}")
    lines.append("THREE-SPLIT STABILITY CHECK")
    lines.append("(Split into thirds. Classify on seg1→seg2 and seg2→seg3. "
                 "'Stable' = same category in both windows.)")
    lines.append(f"{'=' * 160}")

    stable_counts = defaultdict(int)
    unstable = 0
    insufficient = 0
    for _, _, r in all_results:
        sc = r.get("stable_cat")
        if sc is not None:
            stable_counts[sc] += 1
        elif r.get("cat_12") is None or r.get("cat_23") is None:
            insufficient += 1
        else:
            unstable += 1

    lines.append(f"  Stable Cat 1 (nonstationary in both windows): {stable_counts[1]}")
    lines.append(f"  Stable Cat 2 (periodic floor in both windows): {stable_counts[2]}")
    lines.append(f"  Stable Cat 3 (periodic avg in both windows):   {stable_counts[3]}")
    lines.append(f"  Unstable (different category across windows):  {unstable}")
    lines.append(f"  Insufficient data for three-split:             {insufficient}")

    # Detail for stable Cat 2 and 3
    for cat in [2, 3]:
        stable_tables = [(l, t, r) for l, t, r in all_results if r.get("stable_cat") == cat]
        if stable_tables:
            lines.append(f"\n  Stable Cat {cat} tables:")
            for l, t, r in sorted(stable_tables, key=lambda x: -x[2]["y_pct"]):
                lines.append(f"    {l:>20s} / {t:>15s}: Y%={r['y_pct']:.1%}  "
                             f"L1={r['best_error']:.3f}  period={r['winning_period_name'] or '—'}  "
                             f"vol={r['total_skipping_volume']:.0f}")

    output = "\n".join(lines)
    print(output)

    report_path = os.path.join(out_dir, "sort_key_analysis.txt")
    with open(report_path, "w") as f:
        f.write(output)
    print(f"\nWritten: {report_path}")

    # CSV
    csv_rows = []
    for label, table, r in all_results:
        e = r["all_errors"]
        row = {
            "cluster": label,
            "table": table,
            "category": r["category"],
            "best_model": r["best_model"],
            "median_error": r["best_error"],
            "static_error": e.get("static", np.inf),
        }
        for name in CANDIDATE_PERIODS_BINS:
            row[f"floor_{name}_error"] = e.get(f"floor_{name}", np.inf)
            row[f"avg_{name}_error"] = e.get(f"avg_{name}", np.inf)
        row["avg_multi_error"] = e.get("avg_multi", np.inf)
        row.update({
            "winning_period": r["winning_period_name"],
            "y_pct": r["y_pct"],
            "margin_l1": r["margin"],
            "category_regret": r["category_regret"],
            "best_model_regret": r["best_model_regret"],
            "best_regret_value": r["best_regret"],
            "margin_regret": r["margin_regret"],
            "regret_static": r["regrets"].get("static", 0),
            "regret_best_l1": r["regrets"].get(r["best_model"], 0),
            "regret_best_regret": r["best_regret"],
            "regret_static_train": r["regrets_train"].get("static", 0),
            "oracle_benefit": r["oracle_benefit"],
            "stable_cat": r.get("stable_cat"),
            "cat_12": r.get("cat_12"),
            "cat_23": r.get("cat_23"),
            "n_cols": r["n_cols"],
            "n_bins": r["n_bins"],
            "total_skipping_volume": r["total_skipping_volume"],
        })
        csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, "sort_key_analysis.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
