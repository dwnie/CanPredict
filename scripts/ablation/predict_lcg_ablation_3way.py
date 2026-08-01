#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import json
import glob
import csv
import argparse
from typing import List, Dict, Any, Optional

import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hetero_lcg_model import build_model as build_hetero_model


# Determinism

def enable_determinism_on_cpu(seed: int = 42):
    """Enable best-effort deterministic CPU execution and seed random number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# Utilities

def infer_struct_in_dim(g) -> int:
    """Infer struct_x dim D from one graph. Return 0 if missing/invalid."""
    if not hasattr(g, "struct_x"):
        return 0
    sx = getattr(g, "struct_x")
    if not torch.is_tensor(sx):
        return 0
    if sx.dim() == 0:
        return 1
    if sx.dim() == 1:
        return int(sx.size(0))
    if sx.dim() == 2:
        return int(sx.size(1))
    return int(sx.size(-1))


def inverse_transform_y_tensor(y_t: torch.Tensor, transform: str = "none") -> torch.Tensor:
    """Map predictions from transformed target space back to the original scale."""
    if transform == "log1p":
        y_t = torch.clamp(y_t, min=-20.0, max=20.0)
        return torch.expm1(y_t)
    return y_t


def _torch_load_any(path: str, map_location):
    """Load a PyTorch object across versions with different weights_only support."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_list_graphs(obj):
    """Convert a loaded graph container to a list when possible."""
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            return [obj[i] for i in range(len(obj))]
        except Exception:
            return [obj]
    return [obj]


def _safe_build_model(**kwargs):
    """Build a model while retaining compatibility with older model factory signatures."""
    try:
        return build_hetero_model(**kwargs)
    except TypeError:
        kk = dict(kwargs)
        for k in ["gat_heads", "strict_input_dim_check"]:
            if k in kk:
                kk.pop(k)
        return build_hetero_model(**kk)


def load_checkpoints(ckpt_dir: str) -> List[str]:
    """Return all fold checkpoints in a directory."""
    paths = glob.glob(os.path.join(ckpt_dir, "fold_*_best.pth"))
    if not paths:
        raise FileNotFoundError(f"No fold_*_best.pth checkpoints found in {ckpt_dir}")
    paths = sorted(paths)
    print(f"Found {len(paths)} fold checkpoints in {ckpt_dir}:")
    for p in paths:
        print("  -", os.path.basename(p))
    return paths


def build_model_from_ckpt(model_type: str, sample_graph, ckpt: Dict[str, Any], device: torch.device):
    """Build a model from checkpoint metadata and sample-graph dimensions."""
    model_config = ckpt.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}

    # Prefer checkpoint dimensions and fall back to the sample graph.
    pos_in_dim = int(model_config.get("pos_in_dim", sample_graph["pos"].x.size(1)))
    neg_in_dim = int(model_config.get("neg_in_dim", sample_graph["neg"].x.size(1)))
    clause_in_dim = int(model_config.get("clause_in_dim", sample_graph["clause"].x.size(1)))

    hidden_dim = int(model_config.get("hidden_dim", 128))
    num_layers = int(model_config.get("num_layers", 4))
    dropout = float(model_config.get("dropout", 0.1))
    use_struct = bool(model_config.get("use_struct", True))

    struct_in_dim = model_config.get("struct_in_dim", None)
    try:
        struct_in_dim = int(struct_in_dim) if struct_in_dim is not None else 0
    except Exception:
        struct_in_dim = 0
    if use_struct and struct_in_dim <= 0:
        inferred = infer_struct_in_dim(sample_graph)
        struct_in_dim = int(inferred) if inferred > 0 else 15

    gat_heads = int(model_config.get("gat_heads", 4))
    strict_input_dim_check = bool(model_config.get("strict_input_dim_check", False))

    model = _safe_build_model(
        model_type=model_type,
        pos_in_dim=pos_in_dim,
        neg_in_dim=neg_in_dim,
        clause_in_dim=clause_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_struct=use_struct,
        struct_in_dim=struct_in_dim,
        gat_heads=gat_heads,
        strict_input_dim_check=strict_input_dim_check,
    )
    return model.to(device)


@torch.no_grad()
def predict_with_single_ckpt(
    ckpt_path: str,
    model_type: str,
    graphs,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict all supplied graphs with one fold checkpoint."""
    print(f"\n====== Predicting with checkpoint: {os.path.basename(ckpt_path)} ======")
    ckpt = _torch_load_any(ckpt_path, map_location=device)
    if "model_config" not in ckpt or not isinstance(ckpt.get("model_config"), dict):
        ckpt["model_config"] = {}

    target_transform = ckpt.get("target_transform", "log1p")
    label_mean = float(ckpt.get("label_mean", 0.0))
    label_std = float(ckpt.get("label_std", 1.0))
    print(f"  Denormalization parameters: mean={label_mean:.6f}, std={label_std:.6f}, target_transform={target_transform}")

    sample_graph = graphs[0]
    model = build_model_from_ckpt(model_type, sample_graph, ckpt, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean_t = torch.tensor(label_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(label_std, dtype=torch.float32, device=device)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds = []

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)

        pred_t = pred_norm * std_t + mean_t
        pred = inverse_transform_y_tensor(pred_t, target_transform)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)

        preds.extend(pred.detach().cpu().numpy().reshape(-1))

    preds = np.array(preds, dtype=np.float64)
    print(f"  Prediction complete: shape={preds.shape}, pred_range=[{preds.min():.6f}, {preds.max():.6f}]")
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, and relative-error threshold rates."""
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) else 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else 0.0
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

    ae = np.abs(y_true - y_pred) if len(y_true) else np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(y_true == 0, np.nan, ae / np.abs(y_true))

    rel_le_10 = float(np.nanmean(rel_err <= 0.10)) if rel_err.size else 0.0
    rel_le_5 = float(np.nanmean(rel_err <= 0.05)) if rel_err.size else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "rel_le_10_ratio": float(rel_le_10),
        "rel_le_5_ratio": float(rel_le_5),
    }


def aggregate_mean_std(fold_metrics: List[Dict[str, Any]], key: str):
    vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def extract_y_true(graphs) -> np.ndarray:
    ys = []
    for g in graphs:
        y = getattr(g, "y", None)
        if y is None:
            raise ValueError("A graph is missing its y label; prediction errors cannot be evaluated")
        ys.append(float(y.item()) if torch.is_tensor(y) else float(y))
    return np.array(ys, dtype=np.float64)


def write_predictions_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray):
    ae = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(y_true == 0, np.nan, ae / np.abs(y_true))
    rel10 = rel_err <= 0.10
    rel5 = rel_err <= 0.05

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "y_true",
            "y_pred_ensemble",
            "abs_err_ensemble",
            "rel_err_ensemble",
            "rel_le_10_flag",
            "rel_le_5_flag",
        ])
        for i in range(len(y_true)):
            writer.writerow([
                i,
                float(y_true[i]),
                float(y_pred[i]),
                float(ae[i]),
                float(rel_err[i]) if np.isfinite(rel_err[i]) else "",
                int(rel10[i]) if np.isfinite(rel_err[i]) else "",
                int(rel5[i]) if np.isfinite(rel_err[i]) else "",
            ])


# Warm CUDA kernels before timed inference.

@torch.no_grad()
def cuda_warmup(
    ckpt_path: str,
    model_type: str,
    graphs,
    device: torch.device,
    batch_size: int = 32,
):
    """Warm CUDA kernels before timed inference."""
    if device.type != "cuda":
        return
    print("\n[Warmup] Warming up CUDA kernels; this is excluded from inference timing...")
    ckpt = _torch_load_any(ckpt_path, map_location=device)
    model = build_model_from_ckpt(model_type, graphs[0], ckpt, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy_loader = DataLoader(graphs[:min(batch_size, len(graphs))], batch_size=batch_size, shuffle=False)
    dummy_batch = next(iter(dummy_loader)).to(device)
    _ = model(dummy_batch)
    torch.cuda.synchronize()
    del model, ckpt, dummy_batch, dummy_loader
    torch.cuda.empty_cache()
    print("[Warmup] CUDA kernel warmup complete\n")


# Main workflow

def main():
    parser = argparse.ArgumentParser(
        description="Three-way heterogeneous LCG K-fold prediction and evaluation in full-dataset or single-sample mode"
    )
    parser.add_argument("--data_path", type=str, required=True, help="HeteroData .pt dataset to predict, including y labels")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["hetero_sage", "hetero_gin", "hetero_gat"])
    parser.add_argument("--model_dir", type=str, required=True, help="Training output directory containing fold_*_best.pth checkpoints")
    parser.add_argument("--output_dir", type=str, default="./pred_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cpu / cuda / cuda:0 / cuda:1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in full-dataset mode")
    parser.add_argument("--single_index", type=int, default=None,
                        help="Zero-based sample index for single-sample mode. When set, full-dataset inference is skipped.")
    parser.add_argument("--save_single_json", action=argparse.BooleanOptionalAction, default=True,
                        help="Save single_prediction.json in single-sample mode; enabled by default and disabled with --no-save_single_json")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    if args.device == "cpu":
        device = torch.device("cpu")
        enable_determinism_on_cpu(seed=42)
        print("Using device: cpu (deterministic mode enabled)")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} (determinism is not enforced; use --device cpu for strict consistency)")


    print(f"\nLoading dataset: {args.data_path}")
    graphs_obj = _torch_load_any(args.data_path, map_location="cpu")
    graphs_all = _to_list_graphs(graphs_obj)
    if len(graphs_all) == 0:
        raise ValueError("The prediction dataset is empty.")
    print(f"Number of graphs: {len(graphs_all)}")


    s = graphs_all[0]
    struct_dim = infer_struct_in_dim(s)
    print(f"Inferred feature dimensions: pos={int(s['pos'].x.size(1))}, neg={int(s['neg'].x.size(1))}, "
          f"clause={int(s['clause'].x.size(1))}, struct={int(struct_dim)}")
    print("Expected trimmed dimensions for verification: pos=7, neg=8, clause=10, struct=15")

    ckpt_paths = load_checkpoints(args.model_dir)
    n_models = len(ckpt_paths)


    single_payload: Optional[Dict[str, Any]] = None

    if args.single_index is not None:
        # Single-index mode.
        i = int(args.single_index)
        if i < 0 or i >= len(graphs_all):
            raise IndexError(f"single_index is out of range: {i}; expected [0, {len(graphs_all)-1}]")

        graphs = [graphs_all[i]]
        y_true = extract_y_true(graphs)

        print("\n================= Single-sample mode =================")
        print(f"Predicting only the sample at index={i}; full-dataset inference is skipped")
        print("================================================")

        # Warm CUDA kernels before timed inference.
        cuda_warmup(
            ckpt_path=ckpt_paths[0],
            model_type=args.model_type,
            graphs=graphs,
            device=device,
            batch_size=1,
        )

        all_fold_preds = []
        fold_metrics: List[Dict[str, Any]] = []
        per_fold_times = []

        for fold_idx, ckpt_path in enumerate(ckpt_paths):
            fold_start = time.perf_counter()
            preds = predict_with_single_ckpt(
                ckpt_path=ckpt_path,
                model_type=args.model_type,
                graphs=graphs,
                device=device,
                batch_size=1,
            )
            fold_time = time.perf_counter() - fold_start
            per_fold_times.append(fold_time)
            all_fold_preds.append(preds)

            m = compute_metrics(y_true, preds)
            m["fold_index"] = fold_idx
            m["ckpt_name"] = os.path.basename(ckpt_path)
            m["inference_time_sec"] = round(fold_time, 4)
            fold_metrics.append(m)

        total_inference_time = sum(per_fold_times)

        all_fold_preds = np.stack(all_fold_preds, axis=0)  # [K, 1]
        ensemble_pred = all_fold_preds.mean(axis=0)        # [1]
        ensemble_metrics = compute_metrics(y_true, ensemble_pred)
        ensemble_metrics["type"] = "fold_ensemble"

        y_t = float(y_true[0])
        y_p = float(ensemble_pred[0])
        ae = float(abs(y_t - y_p))
        rel_err = float(ae / abs(y_t)) if y_t != 0 else float("nan")
        rel10 = (rel_err <= 0.10) if np.isfinite(rel_err) else None
        rel5 = (rel_err <= 0.05) if np.isfinite(rel_err) else None
        per_fold_pred = [float(all_fold_preds[k, 0]) for k in range(all_fold_preds.shape[0])]

        print("\n================= Single-sample prediction =================")
        print(f"index={i}")
        print(f"y_true={y_t:.6f}")
        print(f"y_pred_ensemble={y_p:.6f}")
        print(f"abs_err={ae:.6f}")
        if np.isfinite(rel_err):
            print(f"rel_err={rel_err:.6f}  (Rel@10%={int(rel10)}, Rel@5%={int(rel5)})")
        else:
            print("rel_err=NaN (y_true==0)")
        print("=================================================")


        print("\n================= Inference timing =================")
        for fi, ft in enumerate(per_fold_times):
            print(f"  [fold {fi}] Inference time: {ft:.3f}s")
        print(f"  Total inference time: {total_inference_time:.3f}s")
        print("=================================================")

        single_payload = {
            "index": i,
            "y_true": y_t,
            "y_pred_ensemble": y_p,
            "abs_err": ae,
            "rel_err": rel_err,
            "rel_le_10_flag": int(rel10) if rel10 is not None else None,
            "rel_le_5_flag": int(rel5) if rel5 is not None else None,
            "per_fold_pred": per_fold_pred,
            "inference_times": {
                "per_fold_seconds": [round(t, 4) for t in per_fold_times],
                "total_seconds": round(total_inference_time, 4),
            },
            "ckpt_names": [m.get("ckpt_name", "") for m in fold_metrics],
        }

        if args.save_single_json:
            single_path = os.path.join(args.output_dir, "single_prediction.json")
            with open(single_path, "w", encoding="utf-8") as f:
                json.dump(single_payload, f, indent=2, ensure_ascii=False)
            print(f"Saved single_prediction.json to {single_path}")


        csv_path = os.path.join(args.output_dir, "predictions.csv")
        write_predictions_csv(csv_path, y_true, ensemble_pred)
        print(f"Saved predictions.csv to {csv_path}")

        # Single-index mode.
        metrics_all = {
            "mode": "single",
            "n_graphs": 1,
            "n_models": int(n_models),
            "model_type": args.model_type,
            "single_index": i,
            "fold_metrics": fold_metrics,
            "ensemble_metrics": ensemble_metrics,
            "single_prediction": single_payload,
            "inference_times": {
                "per_fold_seconds": [round(t, 4) for t in per_fold_times],
                "total_seconds": round(total_inference_time, 4),
            },
        }
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics.json to {metrics_path}")

        print("\nSingle-sample mode complete.")
        return

    # Full-dataset inference mode.
    graphs = graphs_all
    y_true = extract_y_true(graphs)

    print("\n================= Full-dataset mode =================")
    print("Running K-fold batch prediction and evaluation on the complete dataset")
    print("================================================")

    all_fold_preds = []
    fold_metrics: List[Dict[str, Any]] = []
    fold_inference_times = []

    # Warm CUDA kernels before timed inference.
    cuda_warmup(
        ckpt_path=ckpt_paths[0],
        model_type=args.model_type,
        graphs=graphs,
        device=device,
        batch_size=args.batch_size,
    )

    for fold_idx, ckpt_path in enumerate(ckpt_paths):
        fold_start = time.perf_counter()
        preds = predict_with_single_ckpt(
            ckpt_path=ckpt_path,
            model_type=args.model_type,
            graphs=graphs,
            device=device,
            batch_size=args.batch_size,
        )
        fold_time = time.perf_counter() - fold_start
        fold_inference_times.append(fold_time)
        all_fold_preds.append(preds)

        m = compute_metrics(y_true, preds)
        m["fold_index"] = fold_idx
        m["ckpt_name"] = os.path.basename(ckpt_path)
        m["inference_time_sec"] = round(fold_time, 4)
        fold_metrics.append(m)

        print(
            f"  [fold {fold_idx}] MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
            f"R2={m['r2']:.4f}, Rel@10%={m['rel_le_10_ratio']*100:.2f}%, "
            f"Rel@5%={m['rel_le_5_ratio']*100:.2f}%, "
            f"inference_time={fold_time:.3f}s"
        )

    all_fold_preds = np.stack(all_fold_preds, axis=0)  # [K, N]
    ensemble_pred = all_fold_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_true, ensemble_pred)
    ensemble_metrics["type"] = "fold_ensemble"

    total_inference_time = sum(fold_inference_times)

    fold_mean_std = {}
    for key in ["mae", "rmse", "r2", "rel_le_10_ratio", "rel_le_5_ratio"]:
        mean_v, std_v = aggregate_mean_std(fold_metrics, key)
        fold_mean_std[key + "_mean"] = mean_v
        fold_mean_std[key + "_std"] = std_v

    print("\n================= Fold-wise(mean ± std) =================")
    print(f"MAE = {fold_mean_std['mae_mean']:.4f} ± {fold_mean_std['mae_std']:.4f}")
    print(f"RMSE = {fold_mean_std['rmse_mean']:.4f} ± {fold_mean_std['rmse_std']:.4f}")
    print(f"R2 = {fold_mean_std['r2_mean']:.4f} ± {fold_mean_std['r2_std']:.4f}")
    print(f"Rel@10% = {fold_mean_std['rel_le_10_ratio_mean']*100:.2f}% ± {fold_mean_std['rel_le_10_ratio_std']*100:.2f}%")
    print(f"Rel@5%  = {fold_mean_std['rel_le_5_ratio_mean']*100:.2f}% ± {fold_mean_std['rel_le_5_ratio_std']*100:.2f}%")
    print("==========================================================")

    print("\n================= Ensemble aggregate metrics =================")
    print(
        "Ensemble MAE={:.4f}, RMSE={:.4f}, R2={:.4f}, Rel@10%={:.2f}%, Rel@5%={:.2f}%".format(
            ensemble_metrics["mae"],
            ensemble_metrics["rmse"],
            ensemble_metrics["r2"],
            ensemble_metrics["rel_le_10_ratio"] * 100.0,
            ensemble_metrics["rel_le_5_ratio"] * 100.0,
        )
    )
    print("=====================================================")


    print("\n================= Inference timing =================")
    for i, t in enumerate(fold_inference_times):
        print(f"  [fold {i}] Inference time: {t:.3f}s")
    print(f"  Total inference time: {total_inference_time:.3f}s")
    if len(fold_inference_times) > 1:
        ft_mean = np.mean(fold_inference_times)
        ft_std = np.std(fold_inference_times, ddof=1)
        print(f"  Mean inference time: {ft_mean:.3f}s ± {ft_std:.3f}s")
    else:
        print(f"  Mean inference time: {fold_inference_times[0]:.3f}s")
    print("=================================================")


    csv_path = os.path.join(args.output_dir, "predictions.csv")
    write_predictions_csv(csv_path, y_true, ensemble_pred)
    print(f"Saved predictions.csv to {csv_path}")


    ft_arr = np.array(fold_inference_times, dtype=np.float64)

    metrics_all = {
        "mode": "full",
        "n_graphs": int(len(y_true)),
        "n_models": int(n_models),
        "model_type": args.model_type,
        "fold_metrics": fold_metrics,
        "fold_mean_std": fold_mean_std,
        "ensemble_metrics": ensemble_metrics,
        "inference_times": {
            "per_fold_seconds": [round(t, 4) for t in fold_inference_times],
            "total_seconds": round(total_inference_time, 4),
            "mean_seconds": round(float(ft_arr.mean()), 4),
            "std_seconds": round(float(ft_arr.std(ddof=1)), 4) if len(ft_arr) > 1 else 0.0,
        },
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics.json to {metrics_path}")

    print("\nFull-dataset mode complete.")


if __name__ == "__main__":
    main()

"""Usage examples:

Full-dataset inference:
python predict_lcg_ablation_3way.py --data_path DATA.pt --model_type hetero_sage \
  --model_dir CHECKPOINT_DIR --output_dir RESULTS_DIR --device cuda:0

Single-index inference:
python predict_lcg_ablation_3way.py --data_path DATA.pt --model_type hetero_sage \
  --model_dir CHECKPOINT_DIR --output_dir RESULTS_DIR --device cpu --single_index 10"""
