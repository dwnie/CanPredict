#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import glob
import argparse
import warnings
import re
from typing import List, Dict, Any

import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hetero_lcg_model import build_model as build_hetero_model

# Reproducibility
def setup_seed_and_determinism(seed: int, device: torch.device):
    """Seed random generators and enable best-effort determinism on CPU."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device.type == "cpu":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this fully.
            pass


# Utilities
def get_batch_slice_for_index(n: int, index: int, batch_size: int):
    """Return (start, end, offset) for the batch slice that would contain `index`.

    This matches DataLoader(list, batch_size=B, shuffle=False) batching behavior.
    """
    if batch_size <= 0:
        batch_size = 1
    start = (index // batch_size) * batch_size
    end = min(start + batch_size, n)
    offset = index - start
    return start, end, offset

def _safe_nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9):
    if torch.is_tensor(x):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if isinstance(x, np.ndarray):
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return x


def inverse_transform_y_tensor(y_t: torch.Tensor, transform: str = "none") -> torch.Tensor:
    """Map predictions from transformed target space back to the original nonnegative scale."""
    if transform == "log1p":

        y_t = torch.clamp(y_t, min=-20.0, max=20.0)
        pred = torch.expm1(y_t)

        return torch.clamp(pred, min=0.0)
    return y_t


def infer_struct_in_dim(graph) -> int:
    """Infer struct_x input dimension from a (trimmed) HeteroData graph.

    Supports struct_x shaped as:
      - [1, D]  (preferred)
      - [D]
      - scalar
    """
    sx = getattr(graph, "struct_x", None)
    if sx is None:
        print("[WARN] sample_graph.struct_x is missing; fallback struct_in_dim=15")
        return 15
    if not torch.is_tensor(sx):
        raise TypeError(f"struct_x must be a torch.Tensor, got {type(sx)}")
    if sx.dim() == 2:

        if sx.size(0) != 1:
            warnings.warn(
                f"struct_x is 2D but its first dimension is not 1 (shape={tuple(sx.shape)}); "
                f"assuming [1, D] and using the second dimension"
            )
        return int(sx.size(1))
    if sx.dim() == 1:
        return int(sx.numel())
    if sx.dim() == 0:
        return 1
    raise ValueError(f"Unsupported struct_x shape: {tuple(sx.shape)}")


def load_checkpoints(ckpt_dir: str) -> List[str]:
    """Return fold checkpoints from a directory in numeric fold order."""
    paths = glob.glob(os.path.join(ckpt_dir, "fold_*_best.pth"))
    if not paths:
        raise FileNotFoundError(f"No fold_*_best.pth checkpoints found in {ckpt_dir}")
    def _extract_fold_num(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(r"fold_(\d+)_best\.pth", fname)
        return int(m.group(1)) if m else 0

    paths = sorted(paths, key=_extract_fold_num)
    print(f"Found {len(paths)} fold checkpoints in {ckpt_dir}:")
    for p in paths:
        print("  -", os.path.basename(p))
    return paths


def build_model_from_ckpt(
    model_type: str,
    sample_graph,
    ckpt: Dict[str, Any],
    device: torch.device,
):
    """Build a model from checkpoint metadata, falling back to dimensions inferred from a sample graph."""
    model_config = ckpt.get("model_config", {}) or {}


    pos_in_dim = int(model_config.get("pos_in_dim", sample_graph["pos"].x.size(1)))
    neg_in_dim = int(model_config.get("neg_in_dim", sample_graph["neg"].x.size(1)))
    clause_in_dim = int(model_config.get("clause_in_dim", sample_graph["clause"].x.size(1)))

    hidden_dim = int(model_config.get("hidden_dim", 128))
    num_layers = int(model_config.get("num_layers", 4))
    dropout = float(model_config.get("dropout", 0.1))
    use_struct = bool(model_config.get("use_struct", True))
    struct_in_dim = int(model_config.get("struct_in_dim", infer_struct_in_dim(sample_graph)))


    gat_heads = int(model_config.get("gat_heads", 4))
    attn_pool_dropout = float(model_config.get("attn_pool_dropout", 0.1))
    attn_lambda_init = float(model_config.get("attn_lambda_init", 0.1))
    strict_input_dim_check = bool(model_config.get("strict_input_dim_check", True))

    model = build_hetero_model(
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
        attn_pool_dropout=attn_pool_dropout,
        attn_lambda_init=attn_lambda_init,
        strict_input_dim_check=strict_input_dim_check,
    )
    return model.to(device)


# Shared inference path

@torch.no_grad()
def predict_graphs_with_model(
    model,
    graphs,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    target_transform: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict a graph list through the shared DataLoader-based inference path."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_preds = []

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred_norm = _safe_nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)


        pred_t = pred_norm * std_t + mean_t


        pred = inverse_transform_y_tensor(pred_t, target_transform)
        pred = _safe_nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)

        all_preds.extend(pred.detach().cpu().numpy().reshape(-1))

    return np.array(all_preds, dtype=np.float64)


# Single-index inference

@torch.no_grad()
def predict_single_index_with_single_ckpt(
    ckpt_path: str,
    model_type: str,
    graphs,
    device: torch.device,
    single_index: int,
    batch_size: int = 32,
) -> float:
    """Predict one sample through its full-dataset batch slice for consistency."""
    n = len(graphs)
    if single_index < 0 or single_index >= n:
        raise IndexError(f"single_index out of range: {single_index}, n={n}")

    start, end, offset = get_batch_slice_for_index(n, single_index, batch_size)
    slice_graphs = graphs[start:end]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    target_transform = ckpt.get("target_transform", "log1p")
    label_mean = float(ckpt.get("label_mean", 0.0))
    label_std = float(ckpt.get("label_std", 1.0))

    sample_graph = graphs[0]
    model = build_model_from_ckpt(
        model_type=model_type,
        sample_graph=sample_graph,
        ckpt=ckpt,
        device=device,
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model weights; dimensions or hyperparameters may be inconsistent: {e}") from e
    model.eval()

    mean_t = torch.tensor(label_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(label_std, dtype=torch.float32, device=device)

    preds = predict_graphs_with_model(
        model=model,
        graphs=slice_graphs,
        device=device,
        mean_t=mean_t,
        std_t=std_t,
        target_transform=target_transform,
        batch_size=batch_size,
    )
    return float(preds[offset])


# Fold-checkpoint inference

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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    target_transform = ckpt.get("target_transform", "log1p")
    label_mean = float(ckpt.get("label_mean", 0.0))
    label_std = float(ckpt.get("label_std", 1.0))

    print(
        f"Denormalization parameters: mean={label_mean:.4f}, std={label_std:.4f}, "
        f"target_transform={target_transform}"
    )


    sample_graph = graphs[0]
    model = build_model_from_ckpt(
        model_type=model_type,
        sample_graph=sample_graph,
        ckpt=ckpt,
        device=device,
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model weights; dimensions or hyperparameters may be inconsistent: {e}") from e
    model.eval()

    mean_t = torch.tensor(label_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(label_std, dtype=torch.float32, device=device)


    all_preds = predict_graphs_with_model(
        model=model,
        graphs=graphs,
        device=device,
        mean_t=mean_t,
        std_t=std_t,
        target_transform=target_transform,
        batch_size=batch_size,
    )

    print(
        f"Prediction complete: shape={all_preds.shape}, "
        f"pred_range=[{all_preds.min():.2f}, {all_preds.max():.2f}]"
    )
    return all_preds


# Metrics

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, and absolute-error threshold rates."""
    y_true = _safe_nan_to_num(np.asarray(y_true, dtype=np.float64), nan=0.0, posinf=1e9, neginf=0.0)
    y_pred = _safe_nan_to_num(np.asarray(y_pred, dtype=np.float64), nan=0.0, posinf=1e9, neginf=0.0)
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) else 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else 0.0
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    ae = np.abs(y_true - y_pred) if len(y_true) else np.array([])

    ae_le_1 = float((ae <= 1).mean()) if ae.size else 0.0
    ae_le_2 = float((ae <= 2).mean()) if ae.size else 0.0
    ae_le_3 = float((ae <= 3).mean()) if ae.size else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "ae_le_1_ratio": ae_le_1,
        "ae_le_2_ratio": ae_le_2,
        "ae_le_3_ratio": ae_le_3,
    }


def aggregate_mean_std(fold_metrics: List[Dict[str, Any]], key: str):
    """Compute a fold-level mean and sample standard deviation for one metric."""
    vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))


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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(
        model_type=model_type,
        sample_graph=graphs[0],
        ckpt=ckpt,
        device=device,
    )
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
        description="Predict and evaluate a heterogeneous LCG dataset with K-fold models in full-dataset or single-sample mode"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="HeteroData dataset to predict, including ground-truth y values, such as a .pt file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["hetero_sage", "hetero_gin", "hetero_gat"],
        help="Model type used for prediction",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Training output directory containing one or more fold_*_best.pth checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pred_results",
        help="Directory for prediction outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device string, such as cpu, cuda, cuda:0, or cuda:1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Deterministic algorithms are enabled on a best-effort basis only when --device cpu is used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--single_index",
        type=int,
        default=None,
        help="Predict only one sample by graph-list index. The default predicts the full dataset.",
    )

    args = parser.parse_args()


    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    setup_seed_and_determinism(args.seed, device)


    print(f"\nLoading prediction dataset: {args.data_path}")
    graphs = torch.load(args.data_path, map_location="cpu", weights_only=False)

    # Normalize loaded data to a graph list.
    if not isinstance(graphs, list):

        if hasattr(graphs, "__len__") and hasattr(graphs, "__getitem__"):
            try:
                graphs = [graphs[i] for i in range(len(graphs))]
            except Exception as e:
                warnings.warn(f"Failed to convert Dataset to a list: {e}; falling back to a single-graph list")
                graphs = [graphs]
        else:
            graphs = [graphs]

    if len(graphs) == 0:
        raise ValueError("The prediction dataset is empty.")

    print(f"Number of graphs to predict: {len(graphs)}")


    s = graphs[0]
    pos_dim = int(s["pos"].x.size(1))
    neg_dim = int(s["neg"].x.size(1))
    clause_dim = int(s["clause"].x.size(1))
    struct_dim = infer_struct_in_dim(s)
    print(f"Feature dimensions from data: pos={pos_dim}, neg={neg_dim}, clause={clause_dim}, struct={struct_dim}")
    print("Expected trimmed dimensions for verification: pos=7, neg=8, clause=10, struct=15")


    ys = []
    for g in graphs:
        y = getattr(g, "y", None)
        if y is None:
            raise ValueError("A graph is missing its y label; prediction errors cannot be evaluated")
        ys.append(float(y.item()) if torch.is_tensor(y) else float(y))
    y_true_all = np.array(ys, dtype=np.float64)

    print(
        f"Label range [min={y_true_all.min():.2f}, max={y_true_all.max():.2f}], "
        f"mean={y_true_all.mean():.2f}, std={y_true_all.std():.2f}"
    )

    # Load fold checkpoints.
    ckpt_paths = load_checkpoints(args.model_dir)
    n_models = len(ckpt_paths)

    # Single-index mode.
    if args.single_index is not None:
        single_index = int(args.single_index)
        if single_index < 0 or single_index >= len(graphs):
            raise IndexError(f"single_index is out of range: {single_index}, len(graphs)={len(graphs)}")
        y_true = float(y_true_all[single_index])

        print(f"\n[Single-sample mode] Predicting only index={single_index}, y_true={y_true:.4f}")

        # Warm CUDA kernels before timed inference.
        cuda_warmup(
            ckpt_path=ckpt_paths[0],
            model_type=args.model_type,
            graphs=graphs,
            device=device,
            batch_size=args.batch_size,
        )

        per_fold_pred = []
        per_fold_times = []
        for fold_idx, ckpt_path in enumerate(ckpt_paths):
            fold_start = time.perf_counter()
            pred = predict_single_index_with_single_ckpt(
                ckpt_path=ckpt_path,
                model_type=args.model_type,
                graphs=graphs,
                device=device,
                single_index=single_index,
                batch_size=args.batch_size,
            )
            fold_time = time.perf_counter() - fold_start
            per_fold_times.append(fold_time)
            per_fold_pred.append(pred)
            print(f"  [fold {fold_idx}] pred={pred:.10f}, inference_time={fold_time:.3f}s")
        total_inference_time = sum(per_fold_times)


        print(f"  [Inference time] Per fold: {[f'{t:.3f}s' for t in per_fold_times]}")
        print(f"  [Inference time] Total: {total_inference_time:.3f}s")
        ensemble_pred = float(np.mean(per_fold_pred))
        abs_err = abs(ensemble_pred - y_true)
        rel_err = (abs_err / abs(y_true)) if y_true != 0 else np.nan
        print(f"\n[Single-sample ensemble] index={single_index}, y_true={y_true:.10f}, ensemble_pred={ensemble_pred:.10f}, abs_err={abs_err:.6f}, rel_err={rel_err:.6f}")


        os.makedirs(args.output_dir, exist_ok=True)

        single_pred_path = os.path.join(args.output_dir, "single_prediction.json")
        with open(single_pred_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index": single_index,
                    "y_true": y_true,
                    "per_fold_pred": per_fold_pred,
                    "ensemble_pred": ensemble_pred,
                    "abs_err": abs_err,
                    "rel_err": (float(rel_err) if np.isfinite(rel_err) else None),
                    "inference_times": {
                        "per_fold_seconds": [round(t, 4) for t in per_fold_times],
                        "total_seconds": round(total_inference_time, 4),
                    },
                    "model_type": args.model_type,
                    "device": str(device),
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "ckpt_names": [os.path.basename(p) for p in ckpt_paths],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved the single-sample prediction to {single_pred_path}")


        metrics_single = {
            "mode": "single",
            "index": single_index,
            "ckpt_names": [os.path.basename(p) for p in ckpt_paths],
            "per_fold_pred": per_fold_pred,
            "ensemble_pred": ensemble_pred,
            "y_true": y_true,
            "abs_err": abs_err,
            "rel_err": (float(rel_err) if np.isfinite(rel_err) else None),
            "inference_times": {
                "per_fold_seconds": [round(t, 4) for t in per_fold_times],
                "total_seconds": round(total_inference_time, 4),
            },
            "model_type": args.model_type,
            "device": str(device),
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_single, f, indent=2, ensure_ascii=False)
        print(f"Saved single-sample metrics to {metrics_path}")


        import csv
        csv_path = os.path.join(args.output_dir, "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "index",
                "y_true",
                "y_pred_ensemble",
                "abs_err",
                "rel_err",
            ]
            writer.writerow(header)
            writer.writerow([
                single_index,
                y_true,
                ensemble_pred,
                abs_err,
                float(rel_err) if np.isfinite(rel_err) else "NaN",
            ])
        print(f"Saved the single-sample prediction to {csv_path}")
        print("\nSingle-sample prediction complete.")
        return

    # Full-dataset inference mode.
    y_true = y_true_all

    # Warm CUDA kernels before timed inference.
    cuda_warmup(
        ckpt_path=ckpt_paths[0],
        model_type=args.model_type,
        graphs=graphs,
        device=device,
        batch_size=args.batch_size,
    )

    all_fold_preds = []  # [n_models, N_graphs]
    fold_metrics: List[Dict[str, Any]] = []
    fold_inference_times = []

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
            f"R2={m['r2']:.4f}, AE<=1={m['ae_le_1_ratio']*100:.2f}%, "
            f"AE<=2={m['ae_le_2_ratio']*100:.2f}%, "
            f"AE<=3={m['ae_le_3_ratio']*100:.2f}%, "
            f"inference_time={fold_time:.3f}s"
        )

    total_inference_time = sum(fold_inference_times)

    all_fold_preds = np.stack(all_fold_preds, axis=0)  # [n_models, N_graphs]


    fold_mean_std = {}
    for key in ["mae", "rmse", "r2", "ae_le_1_ratio", "ae_le_2_ratio", "ae_le_3_ratio"]:
        mean_v, std_v = aggregate_mean_std(fold_metrics, key)
        fold_mean_std[key + "_mean"] = mean_v
        fold_mean_std[key + "_std"] = std_v


    ensemble_pred = all_fold_preds.mean(axis=0)  # [N_graphs]
    ensemble_metrics = compute_metrics(y_true, ensemble_pred)
    ensemble_metrics["type"] = "fold_ensemble"

    print("\n================= Fold-wise metrics (mean ± std) =================")
    print("MAE = {:.4f} ± {:.4f}".format(fold_mean_std["mae_mean"], fold_mean_std["mae_std"]))
    print("RMSE = {:.4f} ± {:.4f}".format(fold_mean_std["rmse_mean"], fold_mean_std["rmse_std"]))
    print("R2 = {:.4f} ± {:.4f}".format(fold_mean_std["r2_mean"], fold_mean_std["r2_std"]))
    print(
        "AE<=1 ratio = {:.2f}% ± {:.2f}%".format(
            fold_mean_std["ae_le_1_ratio_mean"] * 100.0,
            fold_mean_std["ae_le_1_ratio_std"] * 100.0,
        )
    )
    print(
        "AE<=2 ratio = {:.2f}% ± {:.2f}%".format(
            fold_mean_std["ae_le_2_ratio_mean"] * 100.0,
            fold_mean_std["ae_le_2_ratio_std"] * 100.0,
        )
    )
    print(
        "AE<=3 ratio = {:.2f}% ± {:.2f}%".format(
            fold_mean_std["ae_le_3_ratio_mean"] * 100.0,
            fold_mean_std["ae_le_3_ratio_std"] * 100.0,
        )
    )
    print("===============================================================")

    print("\n================= Ensemble aggregate metrics =================")
    print(
        "Ensemble MAE={:.4f}, RMSE={:.4f}, R2={:.4f}, AE<=1={:.2f}%, AE<=2={:.2f}%, AE<=3={:.2f}%".format(
            ensemble_metrics["mae"],
            ensemble_metrics["rmse"],
            ensemble_metrics["r2"],
            ensemble_metrics["ae_le_1_ratio"] * 100.0,
            ensemble_metrics["ae_le_2_ratio"] * 100.0,
            ensemble_metrics["ae_le_3_ratio"] * 100.0,
        )
    )
    print("===============================================================")


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


    os.makedirs(args.output_dir, exist_ok=True)


    ft_arr = np.array(fold_inference_times, dtype=np.float64)

    metrics_all = {
        "n_graphs": int(len(y_true)),
        "n_models": int(n_models),
        "model_type": args.model_type,
        "feature_dims_from_data": {
            "pos_dim": pos_dim,
            "neg_dim": neg_dim,
            "clause_dim": clause_dim,
            "struct_in_dim": struct_dim,
        },
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
    print(f"Saved metrics to {metrics_path}")


    import csv

    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "index",
            "y_true",
            "y_pred_ensemble",
            "abs_err_ensemble",
            "rel_err_ensemble",
            "ae_le_1",
            "ae_le_2",
            "ae_le_3",
        ]
        writer.writerow(header)

        ensemble_ae = np.abs(y_true - ensemble_pred)
        ensemble_rel_err = np.where(
            y_true == 0, np.nan, ensemble_ae / np.abs(y_true)
        )

        for i in range(len(y_true)):
            ae = float(ensemble_ae[i])
            orig_idx = i
            row = [
                orig_idx,
                float(y_true[i]),
                float(ensemble_pred[i]),
                ae,
                float(ensemble_rel_err[i]) if np.isfinite(ensemble_rel_err[i]) else np.nan,
                1 if ae <= 1.0 else 0,
                1 if ae <= 2.0 else 0,
                1 if ae <= 3.0 else 0,
            ]
            writer.writerow(row)

    print(f"Saved per-graph predictions to {csv_path}")
    print("\nPrediction and evaluation complete.")


if __name__ == "__main__":
    main()

"""Usage examples:

Full-dataset inference:
python predict_hetero_lcg_2way.py --data_path DATA.pt --model_type hetero_sage \
  --model_dir CHECKPOINT_DIR --output_dir RESULTS_DIR --device cuda:0

Single-index inference:
python predict_hetero_lcg_2way.py --data_path DATA.pt --model_type hetero_sage \
  --model_dir CHECKPOINT_DIR --output_dir RESULTS_DIR --device cpu --single_index 7"""
