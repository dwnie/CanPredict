#!/usr/bin/env python3
"""
Usage:
  python train_descriptor_baselines.py \
      [--data_dir ./output] \
      [--output_dir ./output]

Dependencies:
  pip install numpy pandas scikit-learn
  # optional: pip install xgboost  (falls back to RandomForest)
"""

import os
import sys
import argparse
import csv
import math
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

# ── Optional XGBoost / RandomForest ────────────────────────────────────
_XGB_AVAILABLE = False
_RF_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestRegressor
    _RF_AVAILABLE = True
except ImportError:
    pass

if not _XGB_AVAILABLE and not _RF_AVAILABLE:
    print(
        "WARNING: Neither XGBoost nor scikit-learn.ensemble is available.\n"
        "  Install one of:  pip install xgboost   or   pip install scikit-learn\n"
        "  Nonlinear descriptor baseline will be skipped.",
        file=sys.stderr,
    )


# ======================================================================
#  Constants — must match the GNN training protocol
# ======================================================================

FEATURE_NAMES = [
    "log1p_P", "log1p_M", "mean_domain_size", "domain_entropy",
    "log1p_num_clauses", "ratio_FT", "ratio_AMO", "ratio_unit",
    "ratio_cross_FT", "FT_len_mean", "FT_len_max", "clause_len_mean",
    "density_neg2clause", "max_degn2c_norm", "gini_degn2c",
]
N_FEATURES = len(FEATURE_NAMES)

FOLD_SEED_BASE = 42  # matches original training
N_SPLITS = 10
TRAIN_SIZE = 0.8      # 80% train, 20% temp (then 50/50 val/test in GNN)
                        # We only use train vs test; val is not needed.

# Per-coverage label bins (from actual config)
LABEL_BINS_T2 = 8
LABEL_BINS_T3 = 5

TARGET_TRANSFORM = "log1p"


# ======================================================================
#  Stratified split matching the main 2-way training protocol
# ======================================================================

def _build_label_bins_boundaries_np(labels: np.ndarray, label_bins: int) -> np.ndarray:
    """Compute quantile-based bin boundaries in transform space."""
    if label_bins is None or int(label_bins) <= 1:
        return None
    labels_t = np.log1p(np.clip(labels.astype(np.float64), 0.0, None))
    qs = np.quantile(labels_t, np.linspace(0.0, 1.0, int(label_bins) + 1))
    qs = np.unique(qs)
    if qs.size <= 2:
        return None
    return qs[1:-1].astype(np.float64)


def _assign_bin_ids_np(labels: np.ndarray, boundaries_t: np.ndarray) -> np.ndarray:
    """Map labels to bin IDs in transform space."""
    if boundaries_t is None or boundaries_t.size == 0:
        return np.zeros((labels.shape[0],), dtype=np.int64)
    labels_t = np.log1p(np.clip(labels.astype(np.float64), 0.0, None))
    return np.digitize(labels_t, boundaries_t, right=True).astype(np.int64)


def stratified_train_test_split(
    y: np.ndarray, seed: int, label_bins: int
) -> tuple:
    """Create a single stratified 80/10/10 split (returns test indices).

    This reproduces the split logic from split_train_val_test_stratified_by_y
    in the main 2-way training script, but discards the validation set (which is not
    needed by descriptor baselines).

    Returns (train_idx, test_idx) where train_idx is 80% and test_idx is 10%.
    The 10% validation split is folded back into the pool (not used).
    """
    n = len(y)
    idx_all = np.arange(n, dtype=np.int64)

    boundaries = _build_label_bins_boundaries_np(y, label_bins)

    if boundaries is None or boundaries.size == 0:
        # Fall back to random split
        rng = np.random.RandomState(int(seed))
        rng.shuffle(idx_all)
        n_train = int(0.8 * n)
        return idx_all[:n_train].copy(), idx_all[n_train + int(0.1 * n):].copy()

    y_bins = _assign_bin_ids_np(y, boundaries)

    # 80/20 split
    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
        train_rel, temp_rel = next(sss1.split(np.zeros(n), y_bins))
        train_idx = idx_all[train_rel]
        temp_idx = idx_all[temp_rel]
        temp_bins = y_bins[temp_rel]
    except ValueError:
        rng = np.random.RandomState(int(seed))
        rng.shuffle(idx_all)
        n_train = int(0.8 * n)
        n_temp = n - n_train
        train_idx = idx_all[:n_train]
        temp_idx = idx_all[n_train:]
        temp_bins = np.zeros_like(temp_idx)

    # 50/50 split of temp → val/test
    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=int(seed) + 1)
        val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), temp_bins))
        test_idx = temp_idx[test_rel]
    except ValueError:
        rng = np.random.RandomState(int(seed) + 1)
        rng.shuffle(temp_idx)
        half = len(temp_idx) // 2
        test_idx = temp_idx[half:]

    return train_idx.tolist(), test_idx.tolist()


# ======================================================================
#  Data loading
# ======================================================================

def load_data(csv_path: str, label: str) -> tuple:
    """Load a descriptor-baseline CSV.

    Returns (X, y) where:
      - X: np.ndarray of shape [N, 15], descriptors
      - y: np.ndarray of shape [N], labels (original scale)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{label} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[data] Loaded {label}: {len(df)} rows from {csv_path}")

    # Feature columns: feat_0 .. feat_14
    feat_cols = [f"feat_{i}" for i in range(N_FEATURES)]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing feature columns: {missing}")

    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.float64)

    # Quick sanity
    if np.any(np.isnan(X)):
        warnings.warn(f"{label}: X contains NaN values — filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    if np.any(np.isnan(y)):
        raise ValueError(f"{label}: y contains NaN values")

    print(f"[data]   X shape: {X.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")
    return X, y


# ======================================================================
#  Regression models
# ======================================================================

def train_ridge(X_train, y_train_log, alpha=1.0):
    """Train Ridge regression with standard scaling.

    Returns (model, scaler) where model is a fitted Ridge and
    scaler is a fitted StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=float(alpha), random_state=42)
    model.fit(X_scaled, y_train_log)
    return model, scaler


def predict_ridge(model, scaler, X_test):
    """Predict with a trained Ridge model, returns predictions in log space."""
    X_scaled = scaler.transform(X_test)
    return model.predict(X_scaled)


def train_xgb(X_train, y_train_log):
    """Train XGBoost regressor (no scaling needed)."""
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train_log)
    return model


def predict_xgb(model, X_test):
    """Predict with trained XGBoost model."""
    return model.predict(X_test)


def train_rf(X_train, y_train_log):
    """Train RandomForest regressor (fallback when XGBoost unavailable)."""
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)
    return model


def predict_rf(model, X_test):
    """Predict with trained RandomForest model."""
    return model.predict(X_test)


# ======================================================================
#  Metrics (original scale)
# ======================================================================

def compute_metrics(y_true_orig, y_pred_orig):
    """Compute R², MAE, RMSE on the original (inverse-transformed) scale.

    Also returns the number of samples.
    """
    n = len(y_true_orig)
    if n == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": 0}

    # Handle any remaining numerical issues
    y_t = np.asarray(y_true_orig, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred_orig, dtype=np.float64).reshape(-1)

    valid = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[valid]
    y_p = y_p[valid]

    if len(y_t) < 2:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": len(y_t)}

    r2 = r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse), "n": int(len(y_t))}


# ======================================================================
#  10-fold cross-validation
# ======================================================================

def run_10fold_cv(
    X_full, y_full, label_bins, coverage_label, model_name, trainer, predictor
):
    """Run 10-fold stratified CV and return per-fold test metrics + predictions.

    Args:
        X_full:        [N, 15] descriptor matrix
        y_full:        [N] labels (original scale)
        label_bins:    number of label bins for stratification
        coverage_label: "2-way" or "3-way" (for logging)
        model_name:    "Ridge" or "XGBoost"/"RandomForest"
        trainer:       callable (X_train, y_train_log) → model
        predictor:     callable (model, X_test) → predictions (log space)

    Returns:
        fold_results:  list of dicts, one per fold
        all_test_preds: dict mapping test_idx → (y_true, list of fold predictions)
        all_models:    list of (model, scaler_if_applicable) per fold
    """
    n = len(y_full)
    fold_results = []
    all_models = []

    print(f"\n{'=' * 60}")
    print(f"{coverage_label} | {model_name} | 10-fold CV")
    print(f"{'=' * 60}")

    for fold in range(1, N_SPLITS + 1):
        seed = FOLD_SEED_BASE + fold - 1
        train_idx, test_idx = stratified_train_test_split(y_full, seed, label_bins)

        # Train / test split
        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_test = X_full[test_idx]
        y_test = y_full[test_idx]

        # Label transform: log1p
        y_train_log = np.log1p(np.clip(y_train, 0.0, None))

        # Train
        model = trainer(X_train, y_train_log)
        all_models.append(model)

        # Predict (log space → inverse transform)
        pred_log = predictor(model, X_test)
        pred_orig = np.expm1(np.clip(pred_log, None, 20.0))
        pred_orig = np.clip(pred_orig, 0.0, None)

        # Metrics on original scale
        metrics = compute_metrics(y_test, pred_orig)
        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

        print(f"  Fold {fold:2d}: R²={metrics['R2']:.6f}  MAE={metrics['MAE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
              f"n_test={len(test_idx)}")

    # Aggregate
    r2_vals = np.array([r["R2"] for r in fold_results], dtype=np.float64)
    mae_vals = np.array([r["MAE"] for r in fold_results], dtype=np.float64)
    rmse_vals = np.array([r["RMSE"] for r in fold_results], dtype=np.float64)

    valid = np.isfinite(r2_vals)
    if valid.sum() > 0:
        print(f"\n  [{coverage_label} | {model_name}] Fold-wise mean ± std:")
        print(f"    R²:   {r2_vals[valid].mean():.6f} ± {r2_vals[valid].std():.6f}")
        print(f"    MAE:  {mae_vals[valid].mean():.6f} ± {mae_vals[valid].std():.6f}")
        print(f"    RMSE: {rmse_vals[valid].mean():.6f} ± {rmse_vals[valid].std():.6f}")

    return fold_results, all_models


# ======================================================================
#  Target benchmark evaluation (ensemble across 10 folds)
# ======================================================================

def evaluate_target_ensemble(
    X_target, y_target, label_bins, coverage_label, model_name,
    all_models, trainer, predictor
):
    """Train 10 fold-models (same folds as CV) and ensemble-predict targets.

    This matches the GNN protocol: train on each fold's 80% training split,
    predict on all target benchmarks, average the 10 predictions.

    all_models can be None (if we want to re-train) or a list of previously
    trained models from run_10fold_cv.
    """
    print(f"\n[{coverage_label} | {model_name}] Target benchmark ensemble")

    # Load full training data to determine fold splits for re-training
    # (We use the previously trained models stored in all_models)
    n_target = len(y_target)
    if n_target == 0:
        print(f"  WARNING: no target benchmarks — skipping")
        return None

    if all_models is None:
        raise ValueError("all_models must be provided (pre-trained from CV)")

    # Each fold's model predicts on target benchmarks
    fold_preds = np.zeros((len(all_models), n_target), dtype=np.float64)

    for fold_idx, model in enumerate(all_models):
        pred_log = predictor(model, X_target)
        pred_orig = np.expm1(np.clip(pred_log, None, 20.0))
        pred_orig = np.clip(pred_orig, 0.0, None)
        fold_preds[fold_idx, :] = pred_orig

    # Ensemble: average across folds
    ensemble_pred = fold_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_target, ensemble_pred)

    print(f"  Ensemble (n_folds={len(all_models)}): "
          f"R²={ensemble_metrics['R2']:.6f}  "
          f"MAE={ensemble_metrics['MAE']:.6f}  "
          f"RMSE={ensemble_metrics['RMSE']:.6f}")

    per_fold_metrics = []
    for fold_idx in range(len(all_models)):
        m = compute_metrics(y_target, fold_preds[fold_idx])
        per_fold_metrics.append(m)

    return {
        "ensemble": ensemble_metrics,
        "per_fold": per_fold_metrics,
        "fold_preds": fold_preds.tolist(),
    }


# ======================================================================
#  Result storage
# ======================================================================

def _mean_std(vals):
    arr = np.array(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt_val(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def write_results_csv(results_rows: list, csv_path: str):
    """Write all results to CSV."""
    fieldnames = [
        "Coverage", "Dataset", "Model",
        "R2", "MAE", "RMSE", "n",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
    print(f"\n[results] Wrote {csv_path}")


def write_results_tex(results_rows: list, tex_path: str):
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Effect of LCG-based structural representation "
                 r"on covering-array-size prediction. Descriptor-only baselines "
                 r"use graph-level structural statistics without LCG nodes, edges, "
                 r"or message passing.}")
    lines.append(r"  \label{tab:lcg-representation-ablation}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{lllrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Coverage & Dataset & Representation / Model & "
                 r"\(R^2\) & MAE & RMSE \\")
    lines.append(r"    \midrule")

    prev_cov = None
    prev_ds = None
    for row in results_rows:
        cov = row.get("Coverage", "")
        ds = row.get("Dataset", "")
        model = row.get("Model", "")

        # Print coverage row label only on first occurrence
        cov_str = cov if cov != prev_cov else ""
        # Print dataset row label only on first occurrence per coverage
        ds_str = ds if ds != prev_ds or cov != prev_cov else ""

        r2 = _fmt_val(row.get("R2"), 4)
        mae = _fmt_val(row.get("MAE"), 4)
        rmse = _fmt_val(row.get("RMSE"), 4)

        lines.append(f"    {cov_str} & {ds_str} & {model} & {r2} & {mae} & {rmse} \\\\")

        prev_cov = cov
        prev_ds = ds

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[results] Wrote {tex_path}")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Descriptor-only baselines for LCG representation ablation"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./output",
        help="Directory containing CSV files from export_graph_descriptors.py"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory for result CSV and LaTeX files"
    )
    parser.add_argument(
        "--ridge_alpha", type=float, default=1.0,
        help="Ridge regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--no_xgboost", action="store_true", default=False,
        help="Force RandomForest even if XGBoost is available"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect which nonlinear regressor to use
    if args.no_xgboost or not _XGB_AVAILABLE:
        if _RF_AVAILABLE:
            nonlinear_name = "RandomForest"
            nonlinear_trainer = lambda X, y: train_rf(X, y)
            nonlinear_predictor = lambda m, X: predict_rf(m, X)
            print(f"[config] Nonlinear baseline: RandomForest (XGBoost {'disabled' if args.no_xgboost else 'not available'})")
        else:
            nonlinear_name = None
            nonlinear_trainer = None
            nonlinear_predictor = None
            print(f"[config] Nonlinear baseline: UNAVAILABLE (install xgboost or sklearn)")
    else:
        nonlinear_name = "XGBoost"
        nonlinear_trainer = lambda X, y: train_xgb(X, y)
        nonlinear_predictor = lambda m, X: predict_xgb(m, X)
        print(f"[config] Nonlinear baseline: XGBoost")

    # ── Define experiments ────────────────────────────────────────────
    experiments = [
        {
            "coverage": "2-way",
            "label_bins": LABEL_BINS_T2,
            "data_csv": os.path.join(args.data_dir, "descriptor_baseline_t2_data.csv"),
            "target_csv": os.path.join(args.data_dir, "descriptor_baseline_t2_target.csv"),
        },
        {
            "coverage": "3-way",
            "label_bins": LABEL_BINS_T3,
            "data_csv": os.path.join(args.data_dir, "descriptor_baseline_t3_data.csv"),
            "target_csv": os.path.join(args.data_dir, "descriptor_baseline_t3_target.csv"),
        },
    ]

    results_rows = []

    for exp in experiments:
        cov = exp["coverage"]
        label_bins = exp["label_bins"]

        # ── Load data ───────────────────────────────────────────────
        try:
            X_full, y_full = load_data(exp["data_csv"], f"{cov} data")
        except FileNotFoundError as e:
            print(f"\n[SKIP] {cov}: {e}", file=sys.stderr)
            continue

        try:
            X_target, y_target = load_data(exp["target_csv"], f"{cov} target")
        except FileNotFoundError as e:
            print(f"\n[SKIP] {cov} target: {e}", file=sys.stderr)
            X_target, y_target = None, None

        # ── Linear baseline: Ridge ──────────────────────────────────
        print(f"\n{'#' * 60}")
        print(f"# {cov} | Ridge Regression (alpha={args.ridge_alpha})")
        print(f"{'#' * 60}")

        def _ridge_trainer(Xtr, ytr_log):
            return train_ridge(Xtr, ytr_log, alpha=args.ridge_alpha)

        def _ridge_predictor(model, Xte):
            m, s = model  # (Ridge model, StandardScaler)
            return predict_ridge(m, s, Xte)

        ridge_results, ridge_models = run_10fold_cv(
            X_full, y_full, label_bins, cov, "Ridge",
            trainer=_ridge_trainer,
            predictor=_ridge_predictor,
        )
        ridge_test_mean, ridge_test_std = _mean_std([r["R2"] for r in ridge_results])

        results_rows.append({
            "Coverage": cov, "Dataset": "Test",
            "Model": "Global descriptors + Ridge",
            "R2": ridge_test_mean if not math.isnan(ridge_test_mean) else None,
            "MAE": _mean_std([r["MAE"] for r in ridge_results])[0],
            "RMSE": _mean_std([r["RMSE"] for r in ridge_results])[0],
            "n": sum(r["n_test"] for r in ridge_results),
        })
        print(f"  → Test R² = {ridge_test_mean:.6f}")

        if X_target is not None and ridge_models is not None:
            # Re-train Ridge on each fold for target evaluation
            # (use the same models from CV, but Ridge stores (model, scaler))
            def _ridge_target_predictor(model_tuple, Xte):
                m, s = model_tuple
                return predict_ridge(m, s, Xte)

            ridge_target = evaluate_target_ensemble(
                X_target, y_target, label_bins, cov, "Ridge",
                ridge_models,
                trainer=_ridge_trainer,
                predictor=_ridge_target_predictor,
            )
            if ridge_target is not None:
                results_rows.append({
                    "Coverage": cov, "Dataset": "Target",
                    "Model": "Global descriptors + Ridge",
                    "R2": ridge_target["ensemble"]["R2"],
                    "MAE": ridge_target["ensemble"]["MAE"],
                    "RMSE": ridge_target["ensemble"]["RMSE"],
                    "n": len(y_target),
                })
                print(f"  → Target R² = {ridge_target['ensemble']['R2']:.6f}")

        # ── Nonlinear baseline: XGBoost / RandomForest ──────────────
        if nonlinear_name is not None:
            print(f"\n{'#' * 60}")
            print(f"# {cov} | {nonlinear_name}")
            print(f"{'#' * 60}")

            nl_results, nl_models = run_10fold_cv(
                X_full, y_full, label_bins, cov, nonlinear_name,
                trainer=nonlinear_trainer,
                predictor=nonlinear_predictor,
            )
            nl_test_mean, _ = _mean_std([r["R2"] for r in nl_results])

            results_rows.append({
                "Coverage": cov, "Dataset": "Test",
                "Model": f"Global descriptors + {nonlinear_name}",
                "R2": nl_test_mean if not math.isnan(nl_test_mean) else None,
                "MAE": _mean_std([r["MAE"] for r in nl_results])[0],
                "RMSE": _mean_std([r["RMSE"] for r in nl_results])[0],
                "n": sum(r["n_test"] for r in nl_results),
            })
            print(f"  → Test R² = {nl_test_mean:.6f}")

            if X_target is not None and nl_models is not None:
                nl_target = evaluate_target_ensemble(
                    X_target, y_target, label_bins, cov, nonlinear_name,
                    nl_models,
                    trainer=nonlinear_trainer,
                    predictor=nonlinear_predictor,
                )
                if nl_target is not None:
                    results_rows.append({
                        "Coverage": cov, "Dataset": "Target",
                        "Model": f"Global descriptors + {nonlinear_name}",
                        "R2": nl_target["ensemble"]["R2"],
                        "MAE": nl_target["ensemble"]["MAE"],
                        "RMSE": nl_target["ensemble"]["RMSE"],
                        "n": len(y_target),
                    })
                    print(f"  → Target R² = {nl_target['ensemble']['R2']:.6f}")

        # ── Append HeteroSAGE full (from existing results) ──────────
        # These values are from the final prediction results in
        # final_results/predictCAN_model_results.md (standalone predictor,
        # NO affine/offset post-processing).
        # They are hardcoded here as reference — verify before submission.
        if cov == "2-way":
            results_rows.append({
                "Coverage": cov, "Dataset": "Test",
                "Model": "Full heterogeneous LCG + HeteroSAGE",
                "R2": 0.9968, "MAE": 2.5744, "RMSE": 4.9821,
                "n": "~4838",
            })
            results_rows.append({
                "Coverage": cov, "Dataset": "Target",
                "Model": "Full heterogeneous LCG + HeteroSAGE",
                "R2": 0.9991, "MAE": 1.1865, "RMSE": 2.1757,
                "n": 55,
            })
        elif cov == "3-way":
            results_rows.append({
                "Coverage": cov, "Dataset": "Test",
                "Model": "Full heterogeneous LCG + HeteroSAGE",
                "R2": 0.9896, "MAE": 25.6182, "RMSE": 73.1478,
                "n": "~4599",
            })
            results_rows.append({
                "Coverage": cov, "Dataset": "Target",
                "Model": "Full heterogeneous LCG + HeteroSAGE",
                "R2": 0.9995, "MAE": 11.5750, "RMSE": 20.7322,
                "n": 55,
            })

    # ── Save results ──────────────────────────────────────────────────
    csv_out = os.path.join(args.output_dir, "descriptor_baseline_results.csv")
    write_results_csv(results_rows, csv_out)

    tex_out = os.path.join(args.output_dir, "descriptor_baseline_results.tex")
    write_results_tex(results_rows, tex_out)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for row in results_rows:
        cov = row["Coverage"]
        ds = row["Dataset"]
        model = row["Model"]
        r2 = _fmt_val(row.get("R2"), 4)
        mae = _fmt_val(row.get("MAE"), 2)
        rmse = _fmt_val(row.get("RMSE"), 2)
        print(f"  {cov:5s} | {ds:6s} | {model:45s} | R²={r2}  MAE={mae}  RMSE={rmse}")

    print(f"\n[main] Results saved to:")
    print(f"  CSV: {csv_out}")
    print(f"  TeX: {tex_out}")
    print("[main] Done.")


if __name__ == "__main__":
    main()
