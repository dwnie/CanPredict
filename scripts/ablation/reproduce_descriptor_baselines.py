#!/usr/bin/env python3


import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

import joblib
import numpy as np
import pandas as pd

import train_descriptor_baselines as base


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def json_dump(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def train_and_archive(
    X_full,
    y_full,
    label_bins,
    coverage,
    model_name,
    trainer,
    predictor,
    artifacts_dir,
):
    fold_metrics = []
    models = []
    model_slug = model_name.lower()

    for fold in range(1, base.N_SPLITS + 1):
        seed = base.FOLD_SEED_BASE + fold - 1
        train_idx, test_idx = base.stratified_train_test_split(y_full, seed, label_bins)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        split_path = os.path.join(
            artifacts_dir, "splits", coverage, f"fold_{fold:02d}.npz"
        )
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        np.savez_compressed(
            split_path,
            train_idx=train_idx,
            test_idx=test_idx,
            seed=np.asarray([seed], dtype=np.int64),
        )

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_test = X_full[test_idx]
        y_test = y_full[test_idx]
        y_train_log = np.log1p(np.clip(y_train, 0.0, None))

        model = trainer(X_train, y_train_log)
        models.append(model)

        model_dir = os.path.join(
            artifacts_dir, "models", coverage, model_slug, f"fold_{fold:02d}"
        )
        os.makedirs(model_dir, exist_ok=True)
        if model_name == "Ridge":
            ridge_model, scaler = model
            joblib.dump(
                {
                    "model": ridge_model,
                    "scaler": scaler,
                    "feature_names": base.FEATURE_NAMES,
                    "target_transform": base.TARGET_TRANSFORM,
                    "coverage": coverage,
                    "fold": fold,
                    "seed": seed,
                },
                os.path.join(model_dir, "model_and_scaler.joblib"),
            )
        elif model_name == "XGBoost":
            model.save_model(os.path.join(model_dir, "model.json"))
            json_dump(
                {
                    "feature_names": base.FEATURE_NAMES,
                    "target_transform": base.TARGET_TRANSFORM,
                    "coverage": coverage,
                    "fold": fold,
                    "seed": seed,
                    "parameters": model.get_params(),
                },
                os.path.join(model_dir, "metadata.json"),
            )

        pred_log = predictor(model, X_test)
        pred_orig = np.expm1(np.clip(pred_log, None, 20.0))
        pred_orig = np.clip(pred_orig, 0.0, None)
        metrics = base.compute_metrics(y_test, pred_orig)
        fold_metrics.append(
            {
                "Coverage": coverage,
                "Model": model_name,
                "fold": fold,
                "seed": seed,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            }
        )

        pred_path = os.path.join(
            artifacts_dir,
            "predictions",
            "test",
            coverage,
            model_slug,
            f"fold_{fold:02d}.csv",
        )
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pd.DataFrame(
            {
                "sample_index": test_idx,
                "y_true": y_test,
                "y_pred": pred_orig,
                "absolute_error": np.abs(y_test - pred_orig),
            }
        ).to_csv(pred_path, index=False)

        print(
            f"[{coverage} {model_name} fold {fold:02d}] "
            f"R2={metrics['R2']:.12f} MAE={metrics['MAE']:.12f} "
            f"RMSE={metrics['RMSE']:.12f}"
        )

    return fold_metrics, models


def target_predictions(X_target, y_target, models, predictor):
    predictions = []
    for model in models:
        pred_log = predictor(model, X_target)
        pred_orig = np.expm1(np.clip(pred_log, None, 20.0))
        predictions.append(np.clip(pred_orig, 0.0, None))
    fold_predictions = np.asarray(predictions, dtype=np.float64)
    ensemble = fold_predictions.mean(axis=0)
    return ensemble, fold_predictions, base.compute_metrics(y_target, ensemble)


def aggregate_test_row(coverage, model_name, fold_metrics):
    return {
        "Coverage": coverage,
        "Dataset": "Test",
        "Model": f"Global descriptors + {model_name}",
        "R2": float(np.mean([row["R2"] for row in fold_metrics])),
        "MAE": float(np.mean([row["MAE"] for row in fold_metrics])),
        "RMSE": float(np.mean([row["RMSE"] for row in fold_metrics])),
        "n": int(sum(row["n_test"] for row in fold_metrics)),
    }


def compare_metrics(reference_path, reproduced_path, output_path):
    reference = pd.read_csv(reference_path)
    reproduced = pd.read_csv(reproduced_path)
    reference = reference[reference["Model"].str.contains("Ridge|XGBoost", regex=True)]
    keys = ["Coverage", "Dataset", "Model"]
    merged = reference.merge(reproduced, on=keys, suffixes=("_original", "_retrained"))
    for metric in ["R2", "MAE", "RMSE", "n"]:
        merged[f"{metric}_delta"] = (
            pd.to_numeric(merged[f"{metric}_retrained"])
            - pd.to_numeric(merged[f"{metric}_original"])
        )
    merged["metrics_match_1e-12"] = (
        merged[["R2_delta", "MAE_delta", "RMSE_delta"]].abs().max(axis=1) <= 1e-12
    ) & (merged["n_delta"] == 0)
    merged.to_csv(output_path, index=False)
    return merged


def compare_target_predictions(reference_dir, reproduced_dir, output_path):
    rows = []
    for coverage, stem in [("2-way", "t2"), ("3-way", "t3")]:
        original = pd.read_csv(
            os.path.join(reference_dir, f"descriptor_baseline_{stem}_target_predictions.csv")
        )
        reproduced = pd.read_csv(
            os.path.join(reproduced_dir, f"descriptor_baseline_{stem}_target_predictions.csv")
        )
        for model in ["Ridge", "XGBoost"]:
            delta = reproduced[f"{model}_pred"] - original[f"{model}_pred"]
            rows.append(
                {
                    "Coverage": coverage,
                    "Model": model,
                    "n": int(len(delta)),
                    "max_absolute_prediction_delta": float(delta.abs().max()),
                    "mean_absolute_prediction_delta": float(delta.abs().mean()),
                    "predictions_match_1e-12": bool(delta.abs().max() <= 1e-12),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def package_versions():
    names = ["numpy", "pandas", "scikit-learn", "joblib", "xgboost"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    if not base._XGB_AVAILABLE:
        raise RuntimeError("XGBoost is required to reproduce the original experiment")

    os.makedirs(args.artifacts_dir, exist_ok=True)
    metrics_rows = []
    all_fold_metrics = []
    target_outputs = {}

    experiments = [
        ("2-way", "t2", base.LABEL_BINS_T2),
        ("3-way", "t3", base.LABEL_BINS_T3),
    ]
    for coverage, stem, label_bins in experiments:
        data_path = os.path.join(args.data_dir, f"descriptor_baseline_{stem}_data.csv")
        target_path = os.path.join(args.data_dir, f"descriptor_baseline_{stem}_target.csv")
        X_full, y_full = base.load_data(data_path, f"{coverage} data")
        X_target, y_target = base.load_data(target_path, f"{coverage} target")
        target_ids = pd.read_csv(target_path, usecols=["benchmark_id"])["benchmark_id"]

        ridge_trainer = lambda X, y: base.train_ridge(X, y, alpha=args.ridge_alpha)
        ridge_predictor = lambda bundle, X: base.predict_ridge(bundle[0], bundle[1], X)
        ridge_folds, ridge_models = train_and_archive(
            X_full,
            y_full,
            label_bins,
            coverage,
            "Ridge",
            ridge_trainer,
            ridge_predictor,
            args.artifacts_dir,
        )
        all_fold_metrics.extend(ridge_folds)
        metrics_rows.append(aggregate_test_row(coverage, "Ridge", ridge_folds))
        ridge_ensemble, ridge_fold_preds, ridge_target_metrics = target_predictions(
            X_target, y_target, ridge_models, ridge_predictor
        )
        metrics_rows.append(
            {
                "Coverage": coverage,
                "Dataset": "Target",
                "Model": "Global descriptors + Ridge",
                **ridge_target_metrics,
            }
        )

        xgb_folds, xgb_models = train_and_archive(
            X_full,
            y_full,
            label_bins,
            coverage,
            "XGBoost",
            base.train_xgb,
            base.predict_xgb,
            args.artifacts_dir,
        )
        all_fold_metrics.extend(xgb_folds)
        metrics_rows.append(aggregate_test_row(coverage, "XGBoost", xgb_folds))
        xgb_ensemble, xgb_fold_preds, xgb_target_metrics = target_predictions(
            X_target, y_target, xgb_models, base.predict_xgb
        )
        metrics_rows.append(
            {
                "Coverage": coverage,
                "Dataset": "Target",
                "Model": "Global descriptors + XGBoost",
                **xgb_target_metrics,
            }
        )

        target_frame = pd.DataFrame(
            {
                "benchmark_id": target_ids,
                "y_true": y_target,
                "Ridge_pred": ridge_ensemble,
                "Ridge_abs_err": np.abs(y_target - ridge_ensemble),
                "XGBoost_pred": xgb_ensemble,
                "XGBoost_abs_err": np.abs(y_target - xgb_ensemble),
            }
        )
        target_output = os.path.join(
            args.artifacts_dir, "predictions", f"descriptor_baseline_{stem}_target_predictions.csv"
        )
        os.makedirs(os.path.dirname(target_output), exist_ok=True)
        target_frame.to_csv(target_output, index=False)
        np.savez_compressed(
            os.path.join(
                args.artifacts_dir,
                "predictions",
                f"descriptor_baseline_{stem}_target_fold_predictions.npz",
            ),
            ridge=ridge_fold_preds,
            xgboost=xgb_fold_preds,
        )
        target_outputs[coverage] = target_output

    metrics_dir = os.path.join(args.artifacts_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    results_path = os.path.join(metrics_dir, "descriptor_baseline_results_retrained.csv")
    pd.DataFrame(metrics_rows).to_csv(results_path, index=False)
    pd.DataFrame(all_fold_metrics).to_csv(
        os.path.join(metrics_dir, "fold_metrics.csv"), index=False
    )

    metric_comparison = compare_metrics(
        os.path.join(args.reference_dir, "descriptor_baseline_results.csv"),
        results_path,
        os.path.join(metrics_dir, "metrics_comparison.csv"),
    )
    prediction_comparison = compare_target_predictions(
        args.reference_dir,
        os.path.join(args.artifacts_dir, "predictions"),
        os.path.join(metrics_dir, "target_prediction_comparison.csv"),
    )

    input_files = [
        f"descriptor_baseline_{stem}_{kind}.csv"
        for stem in ["t2", "t3"]
        for kind in ["data", "target"]
    ]
    manifest = {
        name: {
            "sha256": sha256_file(os.path.join(args.data_dir, name)),
            "bytes": os.path.getsize(os.path.join(args.data_dir, name)),
        }
        for name in input_files
    }
    json_dump(manifest, os.path.join(args.artifacts_dir, "data_manifest.json"))
    json_dump(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "training": {
                "features": base.FEATURE_NAMES,
                "n_features": base.N_FEATURES,
                "fold_seed_base": base.FOLD_SEED_BASE,
                "n_splits": base.N_SPLITS,
                "train_size": base.TRAIN_SIZE,
                "label_bins_t2": base.LABEL_BINS_T2,
                "label_bins_t3": base.LABEL_BINS_T3,
                "target_transform": base.TARGET_TRANSFORM,
                "ridge_alpha": args.ridge_alpha,
                "xgboost": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                    "verbosity": 0,
                },
            },
        },
        os.path.join(args.artifacts_dir, "run_environment_and_config.json"),
    )
    json_dump(
        {
            "all_metric_rows_match_at_1e-12": bool(
                metric_comparison["metrics_match_1e-12"].all()
            ),
            "all_target_prediction_sets_match_at_1e-12": bool(
                prediction_comparison["predictions_match_1e-12"].all()
            ),
            "maximum_metric_absolute_delta": float(
                metric_comparison[["R2_delta", "MAE_delta", "RMSE_delta"]]
                .abs()
                .to_numpy()
                .max()
            ),
            "maximum_target_prediction_absolute_delta": float(
                prediction_comparison["max_absolute_prediction_delta"].max()
            ),
        },
        os.path.join(metrics_dir, "comparison_summary.json"),
    )

    print("[done] Reproduction artifacts written to", args.artifacts_dir)


if __name__ == "__main__":
    main()
