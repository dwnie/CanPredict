# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore", message="The usage of `scatter")

import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from hetero_lcg_model import build_model


# Structural strata are used only for batch composition.
def extract_features_from_struct_x(graph) -> Dict[str, float]:
    if not hasattr(graph, 'struct_x'):
        raise ValueError(f"LCG graph is missing the struct_x feature: graph.name={getattr(graph, 'name', 'unknown')}")
    struct_x = graph.struct_x
    if torch.is_tensor(struct_x):
        struct_x = struct_x.detach().cpu().numpy()
    else:
        struct_x = np.array(struct_x)

    struct_x = struct_x.flatten()
    if len(struct_x) != 15:
        warnings.warn(f"Invalid struct_x dimension: expected 15, found {len(struct_x)}; using default values")
        return {'log1p_P': 0.0, 'log1p_num_clauses': 0.0, 'gini_deg_n2c': 0.0}

    return {
        'log1p_P': float(struct_x[0]),
        'log1p_num_clauses': float(struct_x[4]),
        'gini_deg_n2c': float(struct_x[14]),
    }


def assign_feature_bins(all_features: List[Dict[str, float]], feature_bins: int = 5) -> Tuple[List[str], Dict]:
    log1p_P_list = np.array([f['log1p_P'] for f in all_features])
    log1p_clauses_list = np.array([f['log1p_num_clauses'] for f in all_features])
    gini_list = np.array([f['gini_deg_n2c'] for f in all_features])

    bins_log1p_P = np.unique(np.quantile(log1p_P_list, np.linspace(0, 1, feature_bins + 1)))
    bins_clauses = np.unique(np.quantile(log1p_clauses_list, np.linspace(0, 1, feature_bins + 1)))
    bins_gini = np.unique(np.quantile(gini_list, np.linspace(0, 1, feature_bins + 1)))

    log1p_P_bin_ids = np.digitize(log1p_P_list, bins_log1p_P[1:-1], right=True)
    clauses_bin_ids = np.digitize(log1p_clauses_list, bins_clauses[1:-1], right=True)
    gini_bin_ids = np.digitize(gini_list, bins_gini[1:-1], right=True)

    strata_keys = [f"P_{p}_C_{c}_G_{g}" for p, c, g in zip(log1p_P_bin_ids, clauses_bin_ids, gini_bin_ids)]

    bin_info = {
        'log1p_P_bins': bins_log1p_P.tolist(),
        'log1p_num_clauses_bins': bins_clauses.tolist(),
        'gini_deg_n2c_bins': bins_gini.tolist(),
        'feature_bins': int(feature_bins),
    }
    return strata_keys, bin_info


# Target transforms
def transform_y_np(y, transform='none'):
    if transform == 'log1p':
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y


def inverse_transform_y_np(y, transform='none'):
    if transform == 'log1p':
        y_inv = np.expm1(y)
        return np.clip(y_inv, a_min=0.0, a_max=None)
    return y


def transform_y_tensor(y_t, transform='none'):
    if transform == 'log1p':
        return torch.log1p(torch.clamp(y_t, min=0.0))
    return y_t


def inverse_transform_y_tensor(y_t, transform='none'):
    if transform == 'log1p':
        y_inv = torch.expm1(y_t)
        return torch.clamp(y_inv, min=0.0)
    return y_t


# Stratification bins
def make_y_bins(graphs: List, label_bins: int, target_transform: str) -> Tuple[np.ndarray, List[float]]:
    ys = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
    y_t = transform_y_np(ys, target_transform)

    qs = np.quantile(y_t, np.linspace(0, 1, label_bins + 1))
    qs = np.unique(qs)
    if len(qs) <= 2:
        return np.zeros_like(y_t, dtype=np.int64), qs.tolist()

    y_bins = np.digitize(y_t, qs[1:-1], right=True).astype(np.int64)
    return y_bins, qs.tolist()


def split_train_val_test_y_stratified(graphs: List, seed: int, label_bins: int, target_transform: str):
    rng = np.random.RandomState(seed)
    n = len(graphs)
    idx_all = np.arange(n)

    y_bins, edges = make_y_bins(graphs, label_bins=label_bins, target_transform=target_transform)

    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, temp_rel = next(sss1.split(idx_all.reshape(-1, 1), y_bins))
    except ValueError:
        rng.shuffle(idx_all)
        cut = int(0.8 * n)
        train_rel = idx_all[:cut]
        temp_rel = idx_all[cut:]

    train_idx = idx_all[train_rel]
    temp_idx = idx_all[temp_rel]

    if len(temp_idx) >= 4:
        temp_bins = y_bins[temp_idx]
        try:
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
            val_rel, test_rel = next(sss2.split(temp_idx.reshape(-1, 1), temp_bins))
            val_idx = temp_idx[val_rel]
            test_idx = temp_idx[test_rel]
        except ValueError:
            rng.shuffle(temp_idx)
            half = len(temp_idx) // 2
            val_idx = temp_idx[:half]
            test_idx = temp_idx[half:]
    else:
        rng.shuffle(temp_idx)
        half = len(temp_idx) // 2
        val_idx = temp_idx[:half]
        test_idx = temp_idx[half:]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), {'y_bin_edges_t': edges}


# Structural strata are used only for batch composition.
class StratifiedBatchSampler(Sampler):
    def __init__(self, graphs: List, batch_size: int, seed: int = 42, key_attr: str = "batch_key"):
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(seed)
        self.key_attr = key_attr

        self.strata2idx: Dict[str, List[int]] = {}
        for idx, g in enumerate(graphs):
            key = getattr(g, self.key_attr, None)
            if key is None:
                key = getattr(g, "strata_key", "unknown")
            self.strata2idx.setdefault(str(key), []).append(idx)

        self.strata_iter: Dict[str, iter] = {}
        for key in self.strata2idx.keys():
            idx_list = self.strata2idx[key].copy()
            self.rng.shuffle(idx_list)
            self.strata_iter[key] = iter(idx_list)

    def __iter__(self):
        strata_keys = list(self.strata_iter.keys())
        if not strata_keys:
            return
        for _ in range(len(self)):
            batch = []
            while len(batch) < self.batch_size:
                for key in strata_keys:
                    if len(batch) >= self.batch_size:
                        break
                    try:
                        idx = next(self.strata_iter[key])
                    except StopIteration:
                        new_idx = self.strata2idx[key].copy()
                        self.rng.shuffle(new_idx)
                        self.strata_iter[key] = iter(new_idx)
                        idx = next(self.strata_iter[key])
                    batch.append(idx)
            yield batch[:self.batch_size]

    def __len__(self):
        total_samples = sum(len(v) for v in self.strata2idx.values())
        return (total_samples + self.batch_size - 1) // self.batch_size


# Early stopping
class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4, warmup=10, verbose=True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, score):
        if epoch <= self.warmup:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            if self.verbose and epoch == self.warmup:
                print(f"  [EarlyStopping] Warmup complete; monitoring begins at epoch={epoch}")
            return False

        if score < self.best_score - self.min_delta:
            if self.verbose:
                delta = self.best_score - score
                print(f"  [EarlyStopping] Metric improved by {delta:.6f} at epoch={epoch}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} consecutive epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  [EarlyStopping] Triggered; best epoch = {self.best_epoch}, best score = {self.best_score:.6f}")
            return False


# Command-line arguments
def parse_args():
    p = argparse.ArgumentParser(description='Heterogeneous GNN regression training on LCGs with Rel@10 optimization and reduced underestimation')

    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='saved_models_hetero')

    p.add_argument('--model_type', type=str, default='hetero_sage',
                   choices=['hetero_sage', 'hetero_gin', 'hetero_gat'])
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--gat_heads', type=int, default=4)
    p.add_argument('--attn_pool_dropout', type=float, default=0.1)
    p.add_argument('--attn_lambda_init', type=float, default=0.1)

    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--weight_decay', type=float, default=1e-5)

    p.add_argument('--n_splits', type=int, default=10)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--use_scheduler', action='store_true', default=False)
    p.add_argument('--scheduler_patience', type=int, default=8)

    p.add_argument('--target_transform', type=str, default='log1p', choices=['none', 'log1p'])


    p.add_argument('--label_bins', type=int, default=4)


    p.add_argument('--feature_bins', type=int, default=5)


    p.add_argument('--rel_huber_delta', type=float, default=0.10, help='Huber delta in relative-error units; 0.10 represents 10%')
    p.add_argument('--rel_denominator_floor', type=float, default=1e-6, help='Relative-error denominator floor: denom=max(y, floor)')
    p.add_argument('--under_weight', type=float, default=2.0, help='Penalty weight for underestimation (pred < y)')
    p.add_argument('--over_weight', type=float, default=1.0, help='Penalty weight for overestimation (pred >= y)')


    p.add_argument('--y_weight_power', type=float, default=0.25, help='Large-target weighting exponent; 0 disables weighting and larger values emphasize large y')
    p.add_argument('--y_weight_clip', type=float, default=5.0, help='Maximum value of w_y')


    p.add_argument('--early_stop_metric', type=str, default='mre', choices=['mae', 'mre'])


    p.add_argument('--enable_affine_calibration', action='store_true', default=True)
    p.add_argument('--calibration_clip_a', type=float, default=2.0)
    p.add_argument('--calibration_clip_b', type=float, default=2.0)

    p.add_argument('--enable_one_sided_offset', action='store_true', default=True)
    p.add_argument('--offset_quantile', type=float, default=0.90, help='Quantile of validation underestimation residuals used as the offset')
    p.add_argument('--offset_max', type=float, default=1e9, help='Maximum offset to limit extreme adjustment')


    p.add_argument('--export_val_preds', action='store_true', default=False)
    p.add_argument('--export_test_preds', action='store_true', default=True)

    p.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--log_interval', type=int, default=10)

    return p.parse_args()


# Utilities
def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    den = torch.clamp(den, min=1e-12)
    return num / den


def relative_huber_per_sample(rel_err_signed: torch.Tensor, delta: float) -> torch.Tensor:
    abs_e = torch.abs(rel_err_signed)
    quad = 0.5 * (abs_e ** 2)
    lin = delta * (abs_e - 0.5 * delta)
    return torch.where(abs_e <= delta, quad, lin)


def compute_norm_params(train_graphs, target_transform='none'):
    labels = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    labels_t = transform_y_np(labels, target_transform)
    mean = float(labels_t.mean())
    std = float(labels_t.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


# Affine calibration
def fit_affine_calibration_in_t(pred_t_np: np.ndarray, y_t_np: np.ndarray,
                                clip_a: float = 2.0, clip_b: float = 2.0) -> Dict[str, float]:
    pred_t_np = np.asarray(pred_t_np).reshape(-1)
    y_t_np = np.asarray(y_t_np).reshape(-1)
    if pred_t_np.size < 2:
        return {'a': 1.0, 'b': 0.0}

    X = np.stack([pred_t_np, np.ones_like(pred_t_np)], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(X, y_t_np, rcond=None)
        a = float(sol[0]); b = float(sol[1])
    except Exception:
        a, b = 1.0, 0.0

    a = float(np.clip(a, -clip_a, clip_a))
    b = float(np.clip(b, -clip_b, clip_b))
    return {'a': a, 'b': b}


def apply_affine_calibration_in_t(pred_t: torch.Tensor, calib: Dict[str, float]) -> torch.Tensor:
    a = float(calib.get('a', 1.0))
    b = float(calib.get('b', 0.0))
    return pred_t * a + b


# One-sided offset calibration
def fit_one_sided_offset(pred_np: np.ndarray, y_np: np.ndarray, q: float = 0.90, offset_max: float = 1e9) -> Dict[str, float]:
    pred_np = np.asarray(pred_np).reshape(-1)
    y_np = np.asarray(y_np).reshape(-1)
    if pred_np.size == 0:
        return {'offset': 0.0}

    resid = y_np - pred_np
    resid_pos = resid[resid > 0]
    if resid_pos.size == 0:
        return {'offset': 0.0}

    q = float(np.clip(q, 0.0, 1.0))
    offset = float(np.quantile(resid_pos, q))
    offset = float(np.clip(offset, 0.0, float(offset_max)))
    return {'offset': offset}


def apply_one_sided_offset(pred: torch.Tensor, offset_dict: Optional[Dict[str, float]]) -> torch.Tensor:
    if offset_dict is None:
        return pred
    off = float(offset_dict.get('offset', 0.0))
    if off <= 0:
        return pred
    return pred + off


# Calibrated inference
@torch.no_grad()
def forward_predict(model, batch, mean_t, std_t, args,
                    affine: Optional[Dict[str, float]] = None,
                    offset: Optional[Dict[str, float]] = None) -> torch.Tensor:
    pred_norm = model(batch)
    pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)

    pred_t_raw = pred_norm * std_t + mean_t
    pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)

    if bool(args.enable_affine_calibration) and affine is not None:
        pred_t_raw = apply_affine_calibration_in_t(pred_t_raw, affine)
        pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)

    pred = inverse_transform_y_tensor(pred_t_raw, args.target_transform)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)

    if bool(args.enable_one_sided_offset) and offset is not None:
        pred = apply_one_sided_offset(pred, offset)

    pred = torch.clamp(pred, min=0.0)
    return pred


# Training and evaluation
def _attach_batch_size(batch):
    if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
        return int(batch.num_graphs)
    return int(batch.y.view(-1).numel())


def train_epoch(model, loader, optimizer, device, args,
                mean_t, std_t, median_y_t: torch.Tensor):
    model.train()
    total_loss = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        bs = _attach_batch_size(batch)

        y = batch.y.view(-1).to(device)

        optimizer.zero_grad()


        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)

        pred_t = pred_norm * std_t + mean_t
        pred_t = torch.clamp(pred_t, min=-20.0, max=20.0)
        pred = inverse_transform_y_tensor(pred_t, args.target_transform)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)
        pred = torch.clamp(pred, min=0.0)

        denom = torch.clamp(y, min=float(args.rel_denominator_floor))
        rel_err_signed = safe_div(pred - y, denom)

        loss_vec = relative_huber_per_sample(rel_err_signed, delta=float(args.rel_huber_delta))


        under_mask = (rel_err_signed < 0).to(loss_vec.dtype)
        w_asym = under_mask * float(args.under_weight) + (1.0 - under_mask) * float(args.over_weight)


        if float(args.y_weight_power) > 0.0:

            y_ratio = safe_div(torch.clamp(y, min=1e-12), torch.clamp(median_y_t, min=1e-12))
            w_y = torch.pow(torch.clamp(y_ratio, min=1e-6), float(args.y_weight_power))
            w_y = torch.clamp(w_y, max=float(args.y_weight_clip))
        else:
            w_y = torch.ones_like(loss_vec)

        loss = (loss_vec * w_asym * w_y).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / total if total > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, args, mean_t, std_t,
             affine: Optional[Dict[str, float]] = None,
             offset: Optional[Dict[str, float]] = None,
             fit_calib: bool = False):
    model.eval()

    total_loss = 0.0
    total = 0

    preds_raw = []
    labels = []

    pred_t_raw_all = []
    y_t_all = []

    for batch in loader:
        batch = batch.to(device)
        bs = _attach_batch_size(batch)

        y = batch.y.view(-1).to(device)

        # Retain uncalibrated predictions for calibration fitting.
        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)

        pred_t_raw = pred_norm * std_t + mean_t
        pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)
        y_t = transform_y_tensor(y, args.target_transform)

        pred_t_raw_all.append(pred_t_raw.detach().cpu())
        y_t_all.append(y_t.detach().cpu())

        pred_raw = inverse_transform_y_tensor(pred_t_raw, args.target_transform)
        pred_raw = torch.nan_to_num(pred_raw, nan=0.0, posinf=1e9, neginf=0.0)
        pred_raw = torch.clamp(pred_raw, min=0.0)


        denom = torch.clamp(y, min=float(args.rel_denominator_floor))
        rel_err_signed = safe_div(pred_raw - y, denom)
        loss_vec = relative_huber_per_sample(rel_err_signed, delta=float(args.rel_huber_delta))

        under_mask = (rel_err_signed < 0).to(loss_vec.dtype)
        w_asym = under_mask * float(args.under_weight) + (1.0 - under_mask) * float(args.over_weight)
        loss_vec = loss_vec * w_asym

        total_loss += loss_vec.sum().item()
        total += bs

        preds_raw.extend(pred_raw.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())

    preds_raw = np.asarray(preds_raw, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)

    affine_out = affine
    offset_out = offset

    # Fit calibration parameters on the current validation loader.
    if fit_calib:
        pred_t_raw_np = torch.cat(pred_t_raw_all, dim=0).numpy().reshape(-1)
        y_t_np = torch.cat(y_t_all, dim=0).numpy().reshape(-1)

        if bool(args.enable_affine_calibration):
            affine_out = fit_affine_calibration_in_t(
                pred_t_raw_np, y_t_np,
                clip_a=float(args.calibration_clip_a),
                clip_b=float(args.calibration_clip_b),
            )


        if bool(args.enable_affine_calibration) and affine_out is not None:
            pred_t_cal = apply_affine_calibration_in_t(torch.from_numpy(pred_t_raw_np).to(torch.float32), affine_out)
            pred_t_cal = torch.clamp(pred_t_cal, min=-20.0, max=20.0)
            pred_cal_np = inverse_transform_y_tensor(pred_t_cal, args.target_transform).numpy().reshape(-1)
        else:
            pred_cal_np = preds_raw.copy()

        pred_cal_np = np.nan_to_num(pred_cal_np, nan=0.0, posinf=1e9, neginf=0.0)
        pred_cal_np = np.clip(pred_cal_np, 0.0, None)

        if bool(args.enable_one_sided_offset):
            offset_out = fit_one_sided_offset(
                pred_cal_np, labels,
                q=float(args.offset_quantile),
                offset_max=float(args.offset_max),
            )

    # Apply affine calibration, inverse transformation, and the one-sided offset.

    preds = preds_raw.copy()

    if bool(args.enable_affine_calibration) and affine_out is not None and preds.size > 0:

        pred_t_raw_np = torch.cat(pred_t_raw_all, dim=0).numpy().reshape(-1)
        pred_t_cal = apply_affine_calibration_in_t(torch.from_numpy(pred_t_raw_np).to(torch.float32), affine_out)
        pred_t_cal = torch.clamp(pred_t_cal, min=-20.0, max=20.0)
        preds = inverse_transform_y_tensor(pred_t_cal, args.target_transform).numpy().reshape(-1)

    preds = np.nan_to_num(preds, nan=0.0, posinf=1e9, neginf=0.0)
    preds = np.clip(preds, 0.0, None)

    if bool(args.enable_one_sided_offset) and offset_out is not None and preds.size > 0:
        preds = preds + float(offset_out.get('offset', 0.0))
        preds = np.clip(preds, 0.0, None)

    avg_loss = total_loss / total if total > 0 else 0.0
    mae = mean_absolute_error(labels, preds) if labels.size else 0.0
    rmse = float(np.sqrt(mean_squared_error(labels, preds))) if labels.size else 0.0
    r2 = float(r2_score(labels, preds)) if labels.size > 1 else 0.0

    denom = np.maximum(labels, float(args.rel_denominator_floor))
    rel_err = np.abs(preds - labels) / denom
    rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=1e9, neginf=0.0)

    rel_stats = {
        'mean_rel': float(rel_err.mean()) if rel_err.size else 0.0,
        'rel_le_10': float((rel_err <= 0.10).mean()) if rel_err.size else 0.0,
        'rel_le_05': float((rel_err <= 0.05).mean()) if rel_err.size else 0.0,
    }

    abs_err = np.abs(preds - labels)
    abs_stats = {
        'within_1': float((abs_err <= 1.0).mean()) if abs_err.size else 0.0,
        'within_2': float((abs_err <= 2.0).mean()) if abs_err.size else 0.0,
        'within_3': float((abs_err <= 3.0).mean()) if abs_err.size else 0.0,
    }

    return avg_loss, r2, float(mae), float(rmse), rel_stats, abs_stats, affine_out, offset_out


# Per-sample prediction collection
@torch.no_grad()
def collect_predictions(model, loader, device, args, mean_t, std_t,
                        affine: Optional[Dict[str, float]] = None,
                        offset: Optional[Dict[str, float]] = None):
    model.eval()

    all_preds = []
    all_labels = []
    all_index = []

    fallback_counter = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).to(device)

        if hasattr(batch, 'global_idx'):
            try:
                idx_t = batch.global_idx.view(-1).detach().cpu().numpy().astype(np.int64)
                idx_list = idx_t.tolist()
            except Exception:
                idx_list = list(range(fallback_counter, fallback_counter + int(y.numel())))
        else:
            idx_list = list(range(fallback_counter, fallback_counter + int(y.numel())))
        fallback_counter += int(y.numel())

        pred = forward_predict(model, batch, mean_t, std_t, args, affine=affine, offset=offset)

        preds_np = pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
        labels_np = y.detach().cpu().numpy().reshape(-1).astype(np.float32)

        all_preds.append(preds_np)
        all_labels.append(labels_np)
        all_index.extend(idx_list)

    if len(all_preds) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    index = np.asarray(all_index, dtype=np.int64)
    return preds, labels, index


# Fold training
def train_one_split(fold, train_graphs, val_graphs, args, device, feature_dims, save_dir):
    mean, std = compute_norm_params(train_graphs, args.target_transform)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    train_labels = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    med_y = float(np.median(train_labels)) if train_labels.size else 1.0
    median_y_t = torch.tensor(med_y, dtype=torch.float32, device=device)

    train_sampler = StratifiedBatchSampler(train_graphs, batch_size=args.batch_size, seed=args.seed + fold, key_attr="batch_key")
    train_loader = DataLoader(train_graphs, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(
        model_type=args.model_type,
        pos_in_dim=feature_dims['pos_dim'],
        neg_in_dim=feature_dims['neg_dim'],
        clause_in_dim=feature_dims['clause_dim'],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attn_pool_dropout=args.attn_pool_dropout,
        attn_lambda_init=args.attn_lambda_init,
        use_struct=True,
        struct_in_dim=feature_dims['struct_dim'],
        gat_heads=args.gat_heads,
        strict_input_dim_check=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-6
        )

    early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta, warmup=args.warmup_epochs, verbose=True)

    best_score = float('inf')
    best = {}
    best_affine = None
    best_offset = None

    history_rows = []

    print("\n" + "=" * 120)
    print(f"Starting fold {fold} training | loss=asym_rel_huber | under_w={args.under_weight} over_w={args.over_weight} "
          f"| offset_q={args.offset_quantile}")
    print("=" * 120)
    print(f"{'Epoch':<6} {'TrainLoss':<10} {'ValLoss':<10} {'ValMRE':<8} {'ValRel@10':<10} "
          f"{'ValMAE':<10} {'ValRMSE':<10} {'ValR2':<8} {'LR':<10} {'Status':<12}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args, mean_t, std_t, median_y_t)

        val_loss, val_r2, val_mae, val_rmse, val_rel, val_abs, affine_out, offset_out = evaluate(
            model, val_loader, device, args, mean_t, std_t, affine=None, offset=None, fit_calib=True
        )

        monitor_score = float(val_mae) if args.early_stop_metric == 'mae' else float(val_rel['mean_rel'])
        improved = early_stop(epoch, monitor_score)

        if scheduler is not None:
            scheduler.step(monitor_score)

        lr = float(optimizer.param_groups[0]['lr'])

        history_rows.append({
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_r2': float(val_r2),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_within_1': float(val_abs['within_1']),
            'val_within_2': float(val_abs['within_2']),
            'val_within_3': float(val_abs['within_3']),
            'val_mre': float(val_rel['mean_rel']),
            'val_rel10': float(val_rel['rel_le_10']),
            'lr': lr,
        })

        if monitor_score < best_score:
            best_score = monitor_score
            best_affine = affine_out
            best_offset = offset_out
            best = {
                'epoch': int(epoch),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_r2': float(val_r2),
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_within_1': float(val_abs['within_1']),
                'val_within_2': float(val_abs['within_2']),
                'val_within_3': float(val_abs['within_3']),
                'val_mre': float(val_rel['mean_rel']),
                'val_rel10': float(val_rel['rel_le_10']),
                'affine': best_affine,
                'offset': best_offset,
                'label_mean': float(mean),
                'label_std': float(std),
                'train_median_y': float(med_y),
            }

            ckpt_path = os.path.join(save_dir, f'fold_{fold}_best.pth')
            torch.save({
                'fold': int(fold),
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_mean': float(mean),
                'label_std': float(std),
                'target_transform': args.target_transform,
                'affine': best_affine,
                'offset': best_offset,
                'best_monitor': float(best_score),
                'best_val_mae': float(val_mae),
                'best_val_mre': float(val_rel['mean_rel']),
                'best_val_rel10': float(val_rel['rel_le_10']),
                'model_config': {
                    'model_type': args.model_type,
                    'pos_in_dim': int(feature_dims['pos_dim']),
                    'neg_in_dim': int(feature_dims['neg_dim']),
                    'clause_in_dim': int(feature_dims['clause_dim']),
                    'hidden_dim': int(args.hidden_dim),
                    'num_layers': int(args.num_layers),
                    'dropout': float(args.dropout),
                    'use_struct': True,
                    'struct_in_dim': int(feature_dims['struct_dim']),
                    'gat_heads': int(args.gat_heads),
                    'attn_pool_dropout': float(args.attn_pool_dropout),
                    'attn_lambda_init': float(args.attn_lambda_init),
                    'strict_input_dim_check': True,
                },
                'train_hparams': {
                    'rel_huber_delta': float(args.rel_huber_delta),
                    'under_weight': float(args.under_weight),
                    'over_weight': float(args.over_weight),
                    'y_weight_power': float(args.y_weight_power),
                    'y_weight_clip': float(args.y_weight_clip),
                    'enable_affine_calibration': bool(args.enable_affine_calibration),
                    'enable_one_sided_offset': bool(args.enable_one_sided_offset),
                    'offset_quantile': float(args.offset_quantile),
                }
            }, ckpt_path)

            status = "✓ Best"
        else:
            status = f"P {early_stop.counter}/{args.patience}"

        if epoch % args.log_interval == 0 or improved or early_stop.early_stop:
            print(f"{epoch:<6} {train_loss:<10.5f} {val_loss:<10.5f} {val_rel['mean_rel']:<8.4f} {val_rel['rel_le_10']:<10.4f} "
                  f"{val_mae:<10.4f} {val_rmse:<10.4f} {val_r2:<8.4f} {lr:<10.2e} {status:<12}")

        if early_stop.early_stop:
            print(f"\n[EarlyStop] epoch={epoch} | best_epoch={early_stop.best_epoch} | best_score={best_score:.6f}")
            break


    hist_df = pd.DataFrame(history_rows)
    hist_path = os.path.join(save_dir, f"fold_{fold}_history.csv")
    hist_df.to_csv(hist_path, index=False)

    print("\n" + "-" * 120)
    print(f"Fold {fold} Best: epoch={best.get('epoch', -1)} | Val MRE={best.get('val_mre', float('nan')):.4f} | "
          f"Val Rel@10={best.get('val_rel10', float('nan')):.4f} | Val MAE={best.get('val_mae', float('nan')):.4f} | "
          f"offset={float((best.get('offset') or {}).get('offset', 0.0)):.4f}")
    print("-" * 120)

    return best


# Command-line entry point
def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{args.model_type}_y_only_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 80)
    print("Starting training with Rel@10 optimization and an underestimation penalty")
    print("=" * 80)
    print(f"save_dir: {save_dir}")
    print(f"device:   {device}")

    graphs = torch.load(args.data_path, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]
    if len(graphs) == 0:
        raise ValueError("The dataset is empty")
    print(f"Loaded {len(graphs)} LCG graphs (HeteroData)")

    # Preserve original indices for per-sample exports.
    for gi, g in enumerate(graphs):
        try:
            g.global_idx = torch.tensor([int(gi)], dtype=torch.long)
        except Exception:
            pass

    # Structural strata are used only for batch composition.
    all_features = []
    missing_struct_x = 0
    for g in graphs:
        try:
            feat = extract_features_from_struct_x(g)
            all_features.append(feat)
        except Exception as e:
            warnings.warn(f"Failed to extract struct_x for graph.name={getattr(g,'name','unknown')}: {e}")
            all_features.append({'log1p_P': 0.0, 'log1p_num_clauses': 0.0, 'gini_deg_n2c': 0.0})
            missing_struct_x += 1

    strata_keys, bin_info = assign_feature_bins(all_features, args.feature_bins)
    for g, key in zip(graphs, strata_keys):
        g.strata_key = key
        g.batch_key = key

    print("\n" + "=" * 80)
    print(f"Completed struct_x binning | feature_bins={args.feature_bins} | missing_struct_x={missing_struct_x}")
    print(f"bin_info: {json.dumps(bin_info, indent=2)}")
    print("=" * 80)

    # Validate feature dimensions.
    s = graphs[0]
    if not hasattr(s, 'struct_x'):
        raise ValueError("The dataset is missing struct_x; verify that the input contains *_with_features.pt training graphs")
    sx = s.struct_x
    if sx.dim() == 2:
        struct_dim = int(sx.shape[1])
    elif sx.dim() == 1:
        struct_dim = int(sx.shape[0])
    elif sx.dim() == 0:
        struct_dim = 1
    else:
        raise ValueError(f"Invalid struct_x dimensions: shape={tuple(sx.shape)}")

    feature_dims = {
        'pos_dim': int(s['pos'].x.shape[1]),
        'neg_dim': int(s['neg'].x.shape[1]),
        'clause_dim': int(s['clause'].x.shape[1]),
        'struct_dim': int(struct_dim),
    }
    print(f"feature_dims: {feature_dims}")

    ys = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
    print(f"label stats: min={ys.min():.2f}, max={ys.max():.2f}, mean={ys.mean():.2f}, std={ys.std():.2f}")

    fold_rows = []

    for fold in range(1, args.n_splits + 1):
        split_seed = args.seed + fold - 1
        train_idx, val_idx, test_idx, split_info = split_train_val_test_y_stratified(
            graphs, seed=split_seed, label_bins=args.label_bins, target_transform=args.target_transform
        )

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        print("\n" + "=" * 80)
        print(f"Fold {fold}/{args.n_splits} | train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")
        print(f"split_info: {split_info}")

        best = train_one_split(fold, train_graphs, val_graphs, args, device, feature_dims, save_dir)

        # Evaluate the best checkpoint using validation-fitted post-processing.
        ckpt_path = os.path.join(save_dir, f'fold_{fold}_best.pth')
        ckpt = torch.load(ckpt_path, map_location=device)

        mc = ckpt.get('model_config', {})
        test_model = build_model(
            model_type=mc.get('model_type', args.model_type),
            pos_in_dim=mc.get('pos_in_dim', feature_dims['pos_dim']),
            neg_in_dim=mc.get('neg_in_dim', feature_dims['neg_dim']),
            clause_in_dim=mc.get('clause_in_dim', feature_dims['clause_dim']),
            hidden_dim=mc.get('hidden_dim', args.hidden_dim),
            num_layers=mc.get('num_layers', args.num_layers),
            dropout=mc.get('dropout', args.dropout),
            attn_pool_dropout=mc.get('attn_pool_dropout', args.attn_pool_dropout),
            attn_lambda_init=mc.get('attn_lambda_init', args.attn_lambda_init),
            use_struct=True,
            struct_in_dim=mc.get('struct_in_dim', feature_dims['struct_dim']),
            gat_heads=mc.get('gat_heads', args.gat_heads),
            strict_input_dim_check=True,
        ).to(device)
        test_model.load_state_dict(ckpt['model_state_dict'])

        mean_t = torch.tensor(float(ckpt['label_mean']), dtype=torch.float32, device=device)
        std_t = torch.tensor(float(ckpt['label_std']), dtype=torch.float32, device=device)
        affine = ckpt.get('affine', None)
        offset = ckpt.get('offset', None)


        if bool(args.export_val_preds):
            val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            val_preds, val_labels, val_index = collect_predictions(test_model, val_loader, device, args, mean_t, std_t, affine=affine, offset=offset)

            v_denom = np.maximum(val_labels, float(args.rel_denominator_floor))
            v_abs_err = np.abs(val_preds - val_labels)
            v_rel_err = v_abs_err / v_denom
            val_df = pd.DataFrame({
                'index': val_index.astype(np.int64),
                'y_true': val_labels.astype(np.float32),
                'y_pred': val_preds.astype(np.float32),
                'abs_err': v_abs_err.astype(np.float32),
                'rel_err': v_rel_err.astype(np.float32),
                'ae_le_1': (v_abs_err <= 1.0).astype(np.int64),
                'ae_le_2': (v_abs_err <= 2.0).astype(np.int64),
                'ae_le_3': (v_abs_err <= 3.0).astype(np.int64),
            }).sort_values('index').reset_index(drop=True)
            val_df.to_csv(os.path.join(save_dir, f"fold_{fold}_val_preds.csv"), index=False)

        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loss, test_r2, test_mae, test_rmse, test_rel, test_abs, _, _ = evaluate(
            test_model, test_loader, device, args, mean_t, std_t, affine=affine, offset=offset, fit_calib=False
        )

        print(f"[Fold {fold}] Test: MRE={test_rel['mean_rel']:.4f} Rel@10={test_rel['rel_le_10']:.4f} "
              f"MAE={test_mae:.4f} RMSE={test_rmse:.4f} R2={test_r2:.4f}")


        if bool(args.export_test_preds):
            test_preds, test_labels, test_index = collect_predictions(test_model, test_loader, device, args, mean_t, std_t, affine=affine, offset=offset)
            denom = np.maximum(test_labels, float(args.rel_denominator_floor))
            abs_err = np.abs(test_preds - test_labels)
            rel_err = abs_err / denom
            out_df = pd.DataFrame({
                'index': test_index.astype(np.int64),
                'y_true': test_labels.astype(np.float32),
                'y_pred': test_preds.astype(np.float32),
                'abs_err': abs_err.astype(np.float32),
                'rel_err': rel_err.astype(np.float32),
                'ae_le_1': (abs_err <= 1.0).astype(np.int64),
                'ae_le_2': (abs_err <= 2.0).astype(np.int64),
                'ae_le_3': (abs_err <= 3.0).astype(np.int64),
            }).sort_values('index').reset_index(drop=True)
            out_df.to_csv(os.path.join(save_dir, f"fold_{fold}_test_preds.csv"), index=False)

        fold_rows.append({
            'fold': int(fold),
            'best_epoch': int(best.get('epoch', -1)),

            'val_r2': float(best.get('val_r2', np.nan)),
            'val_mae': float(best.get('val_mae', np.nan)),
            'val_rmse': float(best.get('val_rmse', np.nan)),
            'val_within_1': float(best.get('val_within_1', np.nan)),
            'val_within_2': float(best.get('val_within_2', np.nan)),
            'val_within_3': float(best.get('val_within_3', np.nan)),
            'val_mre': float(best.get('val_mre', np.nan)),
            'val_rel10': float(best.get('val_rel10', np.nan)),

            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_within_1': float(test_abs['within_1']),
            'test_within_2': float(test_abs['within_2']),
            'test_within_3': float(test_abs['within_3']),
            'test_mre': float(test_rel['mean_rel']),
            'test_rel10': float(test_rel['rel_le_10']),

            'best_offset': float((offset or {}).get('offset', 0.0)),
        })

    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(save_dir, "summary_10fold.csv"), index=False)

    print("\n" + "=" * 90)
    print("10 folds summary saved: summary_10fold.csv")
    print(df.to_string(index=False))
    print("\nMean:")
    print(df.mean(numeric_only=True).to_string())
    print("=" * 90)
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    main()
