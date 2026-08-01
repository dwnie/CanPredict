# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore", message="The usage of `scatter")

import os
import argparse
import json
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from hetero_lcg_model import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Heterogeneous GNN regression training on LCGs with label-only stratification and reduced underestimation')


    p.add_argument('--data_path', type=str, required=True, help='Path to the LCG HeteroData training dataset (.pt)')
    p.add_argument('--output_dir', type=str, default='saved_models_hetero', help='Root directory for saved models')


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
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--weight_decay', type=float, default=1e-5)


    p.add_argument('--target_transform', type=str, default='log1p', choices=['none', 'log1p'])


    p.add_argument('--loss_type', type=str, default='rel_asym_huber',
                   choices=['huber_norm', 'rel_asym_huber'],
                   help='huber_norm: Huber loss in transformed and normalized space; rel_asym_huber: relative-error Huber loss in raw space with underestimation weighting')
    p.add_argument('--huber_delta', type=float, default=1.0, help='Delta used by huber_norm')

    p.add_argument('--rel_huber_delta', type=float, default=0.10,
                   help='Delta used by rel_asym_huber in relative-error units; 0.10 represents 10%')
    p.add_argument('--rel_denominator_floor', type=float, default=1e-6,
                   help='Relative-error denominator: max(y, floor)')
    p.add_argument('--under_weight', type=float, default=2.5,
                   help='Penalty multiplier for underestimation (pred < y); values above 1 reduce underestimation')
    p.add_argument('--over_weight', type=float, default=1.0,
                   help='Penalty multiplier for overestimation (pred >= y), typically 1')


    p.add_argument('--y_weight_power', type=float, default=0.25,
                   help='Additional weighting exponent for large y values; 0 disables weighting and 0.25 to 0.5 is typical')
    p.add_argument('--y_weight_clip', type=float, default=5.0,
                   help='Maximum multiplier for y-based weighting to prevent excessive weights')


    p.add_argument('--n_splits', type=int, default=10)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--use_scheduler', action='store_true', default=False)
    p.add_argument('--scheduler_patience', type=int, default=8)

    # Select the early-stopping metric.
    p.add_argument('--early_stop_metric', type=str, default='mre', choices=['mae', 'mre', 'under_rate'])


    p.add_argument('--label_bins', type=int, default=8, help='Number of y bins for stratified splitting, sampling, and inverse-frequency weighting')


    p.add_argument('--no_label_weight', action='store_true', default=False,
                   help='Disable inverse-frequency weighting by label bin; enabled by default')
    p.add_argument('--label_weight_power', type=float, default=1.0,
                   help='Inverse-frequency weight exponent for label bins; 1.0 is linear and values above 1 emphasize rare bins')


    p.add_argument('--tail_quantile', type=float, default=0.90,
                   help='Training-set quantile threshold used to define tail samples, in [0, 1]')
    p.add_argument('--tail_min_y', type=float, default=50.0,
                   help='Minimum raw y value for the tail threshold')
    p.add_argument('--tail_boost', type=float, default=2.0,
                   help='Additional weight multiplier for tail samples, at least 1')


    p.add_argument('--small_y', type=float, default=20.0, help='Small-target threshold: y <= small_y')
    p.add_argument('--small_boost', type=float, default=1.3, help='Additional weight multiplier for small-target samples, at least 1')

    # Post-processing
    p.add_argument('--enable_calibration', action='store_true', default=True,
                   help='Fit affine calibration parameters a and b in transformed space on the validation set')
    p.add_argument('--calibration_clip_a', type=float, default=2.0)
    p.add_argument('--calibration_clip_b', type=float, default=2.0)

    p.add_argument('--enable_one_sided_offset', action='store_true', default=True,
                   help='Fit a nonnegative upward offset on the validation set')
    p.add_argument('--offset_quantile', type=float, default=0.90,
                   help='offset = quantile(max(0, y - pred_cal), q)')
    p.add_argument('--offset_max', type=float, default=50.0,
                   help='Maximum offset to prevent excessive upward adjustment')


    p.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_interval', type=int, default=10)

    return p.parse_args()


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


# Normalization
def compute_normalization_params(train_graphs, target_transform='none'):
    labels = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    labels_t = transform_y_np(labels, target_transform)
    mean = float(labels_t.mean()) if labels_t.size else 0.0
    std = float(labels_t.std()) if labels_t.size else 1.0
    if std < 1e-8:
        print(" Label standard deviation is near zero; setting std to 1.0")
        std = 1.0
    return mean, std


# Stratification bins
def _build_label_bins_boundaries_np(labels: np.ndarray, label_bins: int, target_transform: str) -> Optional[np.ndarray]:
    if label_bins is None or int(label_bins) <= 1:
        return None
    labels_t = transform_y_np(labels.astype(np.float32), target_transform)
    qs = np.quantile(labels_t, np.linspace(0.0, 1.0, int(label_bins) + 1))
    qs = np.unique(qs)
    if qs.size <= 2:
        return None
    return qs[1:-1].astype(np.float32)


def _assign_bin_ids_np(labels: np.ndarray, boundaries_t: Optional[np.ndarray], target_transform: str) -> np.ndarray:
    if boundaries_t is None or boundaries_t.size == 0:
        return np.zeros((labels.shape[0],), dtype=np.int64)
    labels_t = transform_y_np(labels.astype(np.float32), target_transform)
    return np.digitize(labels_t, boundaries_t, right=True).astype(np.int64)


# Label-bin weighting
def compute_label_weighting(train_graphs: List,
                            label_bins: int,
                            target_transform: str,
                            label_weight_power: float = 1.0,
                            tail_quantile: float = 0.90,
                            tail_min_y: float = 50.0):
    labels = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    if labels.size == 0:
        return None, None, float('inf')

    qy = float(np.quantile(labels, np.clip(tail_quantile, 0.0, 1.0)))
    tail_thr = max(qy, float(tail_min_y))

    if label_bins is None or int(label_bins) <= 1:
        return None, None, tail_thr

    boundaries_np = _build_label_bins_boundaries_np(labels, int(label_bins), target_transform)
    if boundaries_np is None:
        return None, None, tail_thr

    bin_ids = _assign_bin_ids_np(labels, boundaries_np, target_transform)
    n_bins_eff = int(boundaries_np.size + 1)
    counts = np.bincount(bin_ids, minlength=n_bins_eff).astype(np.float32)
    counts[counts <= 0] = 1.0

    inv = (counts.sum() / counts).astype(np.float32)
    p = float(label_weight_power)
    if p < 0:
        p = 0.0
    if p != 1.0:
        inv = np.power(inv, p)

    inv = inv / max(inv.mean(), 1e-6)

    label_boundaries_t = torch.tensor(boundaries_np, dtype=torch.float32)
    label_bin_weights = torch.tensor(inv, dtype=torch.float32)
    return label_boundaries_t, label_bin_weights, tail_thr


# Batch sampling
class LabelStratifiedBatchSampler(Sampler):
    def __init__(self,
                 graphs: List,
                 batch_size: int,
                 seed: int = 42,
                 label_bins: int = 8,
                 target_transform: str = 'log1p'):
        self.graphs = graphs
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(int(seed))

        labels = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
        boundaries_np = _build_label_bins_boundaries_np(labels, int(label_bins), target_transform)

        if boundaries_np is None:
            self.boundaries_np = None
            self.bin2idx = None
            self.indices = np.arange(len(graphs), dtype=np.int64)
            self.rng.shuffle(self.indices)
            return

        self.boundaries_np = boundaries_np
        bin_ids = _assign_bin_ids_np(labels, boundaries_np, target_transform)

        self.bin2idx: Dict[int, List[int]] = {}
        for i, b in enumerate(bin_ids.tolist()):
            self.bin2idx.setdefault(int(b), []).append(i)

        self._reset_iters()

    def _reset_iters(self):
        self.bin_iters: Dict[int, iter] = {}
        if self.bin2idx is None:
            return
        for b, idxs in self.bin2idx.items():
            idxs = idxs.copy()
            self.rng.shuffle(idxs)
            self.bin_iters[b] = iter(idxs)

    def __iter__(self):
        n = len(self.graphs)
        if n == 0:
            return

        if self.bin2idx is None:
            idxs = self.indices.copy()
            self.rng.shuffle(idxs)
            for i in range(0, n, self.batch_size):
                yield idxs[i:i + self.batch_size].tolist()
            return

        bins = list(self.bin_iters.keys())
        if len(bins) == 0:
            return

        for _ in range(len(self)):
            batch = []
            while len(batch) < self.batch_size:
                for b in bins:
                    if len(batch) >= self.batch_size:
                        break
                    try:
                        idx = next(self.bin_iters[b])
                    except StopIteration:
                        self._reset_iters()
                        idx = next(self.bin_iters[b])
                    batch.append(idx)
            yield batch[:self.batch_size]

    def __len__(self):
        n = len(self.graphs)
        return (n + self.batch_size - 1) // self.batch_size


# Early stopping
class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4, warmup=10, verbose=False):
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
                return True
            return False

        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'warmup': self.warmup,
        }


# Dataset splitting
def split_train_val_test_stratified_by_y(graphs: List,
                                         seed: int = 42,
                                         label_bins: int = 8,
                                         target_transform: str = 'log1p'):
    rng = np.random.RandomState(int(seed))
    n = len(graphs)
    idx_all = np.arange(n, dtype=np.int64)
    if n == 0:
        return [], [], []

    labels = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
    boundaries_np = _build_label_bins_boundaries_np(labels, int(label_bins), target_transform)
    if boundaries_np is None:
        rng.shuffle(idx_all)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        return idx_all[:n_train].tolist(), idx_all[n_train:n_train + n_val].tolist(), idx_all[n_train + n_val:].tolist()

    y_bins = _assign_bin_ids_np(labels, boundaries_np, target_transform)

    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
        train_rel, temp_rel = next(sss1.split(np.zeros(n), y_bins))
        train_idx = idx_all[train_rel]
        temp_idx = idx_all[temp_rel]
        temp_bins = y_bins[temp_rel]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=int(seed) + 1)
        val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), temp_bins))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    except ValueError:
        rng.shuffle(idx_all)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        return idx_all[:n_train].tolist(), idx_all[n_train:n_train + n_val].tolist(), idx_all[n_train + n_val:].tolist()


# Utilities
def _attach_batch_size(batch):
    if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
        bs = int(batch.num_graphs)
    else:
        bs = int(batch.y.view(-1).numel())
    batch.batch_size = bs
    return bs


def _relative_huber(rel_err_signed: torch.Tensor, delta: float) -> torch.Tensor:
    abs_e = torch.abs(rel_err_signed)
    quad = 0.5 * (abs_e ** 2)
    lin = delta * (abs_e - 0.5 * delta)
    return torch.where(abs_e <= delta, quad, lin)


def compute_rel_metrics(labels: np.ndarray, preds: np.ndarray, floor: float = 1e-6) -> Dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.size == 0:
        return {'mre': 0.0, 'rel10': 0.0, 'under_rate': 0.0}

    denom = np.maximum(labels, float(floor))
    rel = np.abs(preds - labels) / denom
    rel = np.nan_to_num(rel, nan=0.0, posinf=1e9, neginf=0.0)
    under = (preds < labels).astype(np.float32)

    return {
        'mre': float(rel.mean()),
        'rel10': float((rel <= 0.10).mean()),
        'under_rate': float(under.mean()),
    }


def fit_affine_calibration_in_t(pred_t_np: np.ndarray,
                                y_t_np: np.ndarray,
                                clip_a: float = 2.0,
                                clip_b: float = 2.0) -> Dict[str, float]:
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


def apply_postprocess(pred_t_raw: torch.Tensor,
                      target_transform: str,
                      calib: Optional[Dict[str, float]] = None,
                      offset: float = 0.0) -> torch.Tensor:
    pred_t = pred_t_raw
    if calib is not None:
        a = float(calib.get('a', 1.0))
        b = float(calib.get('b', 0.0))
        pred_t = pred_t * a + b

    pred_t = torch.clamp(pred_t, min=-20.0, max=20.0)
    pred = inverse_transform_y_tensor(pred_t, target_transform)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)

    if float(offset) > 0:
        pred = pred + float(offset)
    pred = torch.clamp(pred, min=0.0)
    return pred


@torch.no_grad()
def collect_val_pred_t_and_y_t(model, loader, device, mean_t, std_t, target_transform: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    pt = []
    yt = []
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).to(device)
        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6).view(-1)

        pred_t_raw = pred_norm * std_t + mean_t
        pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)
        y_t = transform_y_tensor(y, target_transform)

        pt.append(pred_t_raw.detach().cpu().numpy().reshape(-1))
        yt.append(y_t.detach().cpu().numpy().reshape(-1))

    if len(pt) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(pt).astype(np.float32), np.concatenate(yt).astype(np.float32)


def fit_postproc_on_val(model, val_loader, device, mean_t, std_t, args) -> Dict:
    postproc = {'calib': None, 'offset': 0.0}

    pred_t_raw_np, y_t_np = collect_val_pred_t_and_y_t(
        model, val_loader, device, mean_t, std_t, args.target_transform
    )
    if pred_t_raw_np.size == 0:
        return postproc

    calib = None
    if bool(args.enable_calibration):
        calib = fit_affine_calibration_in_t(
            pred_t_raw_np, y_t_np,
            clip_a=float(args.calibration_clip_a),
            clip_b=float(args.calibration_clip_b),
        )

    pred_t_raw_t = torch.from_numpy(pred_t_raw_np).to(torch.float32)
    pred_cal = apply_postprocess(pred_t_raw_t, args.target_transform, calib=calib, offset=0.0).cpu().numpy().reshape(-1)
    y_raw = inverse_transform_y_np(y_t_np, args.target_transform).reshape(-1)

    offset = 0.0
    if bool(args.enable_one_sided_offset):
        resid_pos = np.maximum(0.0, y_raw - pred_cal)
        if resid_pos.size > 0:
            q = float(np.clip(args.offset_quantile, 0.0, 1.0))
            offset = float(np.quantile(resid_pos, q))
            offset = float(np.clip(offset, 0.0, float(args.offset_max)))

    postproc['calib'] = calib
    postproc['offset'] = offset
    return postproc


# Training and evaluation
def train_epoch(model,
                loader,
                criterion,
                optimizer,
                device,
                mean_t,
                std_t,
                args,
                label_boundaries_t=None,
                label_bin_weights=None,
                tail_threshold_y=None):
    """Run one optimization epoch and report both the optimization loss and normalized-space Huber loss."""
    model.train()
    total_loss_main = 0.0
    total_loss_norm = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        bs = _attach_batch_size(batch)

        y = batch.y.view(-1).to(device)

        optimizer.zero_grad()
        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6).view(-1)

        # Track normalized-space Huber loss for reporting.
        with torch.no_grad():
            y_t = transform_y_tensor(y, args.target_transform)
            y_norm = (y_t - mean_t) / std_t
            loss_norm_vec = criterion(pred_norm, y_norm)  # reduction='none'
            total_loss_norm += float(loss_norm_vec.sum().item())

        # Combine label-bin, small-target, and tail weights.
        w = torch.ones_like(y, dtype=torch.float32, device=device)

        if (not args.no_label_weight) and (label_boundaries_t is not None) and (label_bin_weights is not None):
            y_t = transform_y_tensor(y, args.target_transform)
            boundaries = label_boundaries_t.to(device)
            bin_w = label_bin_weights.to(device)
            bin_ids = torch.bucketize(y_t.detach(), boundaries, right=True)
            w = w * bin_w[bin_ids]

        if float(args.small_boost) > 1.0 and float(args.small_y) > 0:
            small_mask = (y.detach() <= float(args.small_y))
            w = torch.where(small_mask, w * float(args.small_boost), w)

        if (tail_threshold_y is not None) and np.isfinite(float(tail_threshold_y)) and float(args.tail_boost) > 1.0:
            tail_mask = (y.detach() >= float(tail_threshold_y))
            w = torch.where(tail_mask, w * float(args.tail_boost), w)

        w = w / w.mean().clamp_min(1e-6)

        # Optimization loss.
        if args.loss_type == 'rel_asym_huber':
            pred_t_raw = pred_norm * std_t + mean_t
            pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)
            pred = inverse_transform_y_tensor(pred_t_raw, args.target_transform)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)

            denom = torch.clamp(y, min=float(args.rel_denominator_floor))
            rel_err_signed = (pred - y) / denom

            loss_vec = _relative_huber(rel_err_signed, delta=float(args.rel_huber_delta))

            asym = torch.where(pred < y, float(args.under_weight), float(args.over_weight))
            loss_vec = loss_vec * asym

            if float(args.y_weight_power) > 0:
                yw = torch.pow(torch.clamp(y, min=1.0), float(args.y_weight_power))
                yw = yw / yw.mean().clamp_min(1e-6)
                yw = torch.clamp(yw, max=float(args.y_weight_clip))
                loss_vec = loss_vec * yw
        else:
            y_t = transform_y_tensor(y, args.target_transform)
            y_norm = (y_t - mean_t) / std_t
            loss_vec = criterion(pred_norm, y_norm)

        loss_main = (loss_vec * w).mean()
        loss_main.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss_main += float(loss_main.item()) * bs
        total += bs

    return {
        'loss_main': (total_loss_main / total if total > 0 else 0.0),
        'loss_norm': (total_loss_norm / total if total > 0 else 0.0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, mean_t, std_t, args, postproc: Optional[Dict] = None):
    """Evaluate normalized-space loss and original-scale regression metrics with optional post-processing."""
    model.eval()
    preds = []
    labels = []
    total_loss_norm = 0.0
    total = 0

    calib = None
    offset = 0.0
    if postproc is not None:
        calib = postproc.get('calib', None)
        offset = float(postproc.get('offset', 0.0))

    for batch in loader:
        batch = batch.to(device)
        bs = _attach_batch_size(batch)

        y = batch.y.view(-1).to(device)
        pred_norm = model(batch)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6).view(-1)

        # 1) norm-space loss (Huber on normalized target)
        y_t = transform_y_tensor(y, args.target_transform)
        y_norm = (y_t - mean_t) / std_t
        loss_vec_norm = criterion(pred_norm, y_norm)  # reduction='none'
        total_loss_norm += float(loss_vec_norm.sum().item())

        # 2) raw-space prediction for metrics
        pred_t_raw = pred_norm * std_t + mean_t
        pred_t_raw = torch.clamp(pred_t_raw, min=-20.0, max=20.0)
        pred = apply_postprocess(pred_t_raw, args.target_transform, calib=calib, offset=offset)

        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())

        total += bs

    preds = np.array(preds, dtype=np.float32).reshape(-1)
    labels = np.array(labels, dtype=np.float32).reshape(-1)

    avg_loss_norm = total_loss_norm / total if total > 0 else 0.0
    mae = mean_absolute_error(labels, preds) if labels.size else 0.0
    rmse = float(np.sqrt(mean_squared_error(labels, preds))) if labels.size else 0.0
    r2 = float(r2_score(labels, preds)) if labels.size > 1 else 0.0

    abs_err = np.abs(labels - preds) if labels.size else np.array([], dtype=np.float32)
    within_1 = float((abs_err <= 1.0).mean()) if abs_err.size else 0.0
    within_2 = float((abs_err <= 2.0).mean()) if abs_err.size else 0.0
    within_3 = float((abs_err <= 3.0).mean()) if abs_err.size else 0.0

    relm = compute_rel_metrics(labels, preds, floor=float(args.rel_denominator_floor))

    return {
        'loss': float(avg_loss_norm),
        'loss_norm': float(avg_loss_norm),
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'within_1': float(within_1),
        'within_2': float(within_2),
        'within_3': float(within_3),
        'mre': float(relm['mre']),
        'rel10': float(relm['rel10']),
        'under_rate': float(relm['under_rate']),
    }


# Fold training
def train_fold(fold, model, train_graphs, val_graphs, args, device, mean, std, save_dir, feature_dims):
    label_boundaries_t, label_bin_weights, tail_threshold_y = compute_label_weighting(
        train_graphs=train_graphs,
        label_bins=args.label_bins,
        target_transform=args.target_transform,
        label_weight_power=args.label_weight_power,
        tail_quantile=args.tail_quantile,
        tail_min_y=args.tail_min_y,
    )

    criterion = nn.HuberLoss(delta=args.huber_delta, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    mon_ema = None
    ema_alpha = 0.2
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=args.scheduler_patience, min_lr=1e-6
        )

    early_stop = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        warmup=args.warmup_epochs,
        verbose=False,
    )

    mean_t = torch.tensor(float(mean), dtype=torch.float32, device=device)
    std_t = torch.tensor(float(std), dtype=torch.float32, device=device)

    train_sampler = LabelStratifiedBatchSampler(
        train_graphs,
        batch_size=args.batch_size,
        seed=args.seed + fold,
        label_bins=args.label_bins,
        target_transform=args.target_transform,
    )
    train_loader = DataLoader(train_graphs, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best = {}
    best_state = None

    history = {
        'epoch': [],
        'train_loss_norm': [],
        'val_loss_norm': [],
        'val_r2': [], 'val_mae': [], 'val_rmse': [],
        'val_within_1': [], 'val_within_2': [], 'val_within_3': [],
        'val_mre': [], 'val_rel10': [], 'val_under_rate': [],
        'lr': [],
        'monitor': []
    }


    print("\n" + "-" * 118)
    print(f"Training fold {fold}")
    print("-" * 118)
    print(f"{'Epoch':<6} {'Train_Loss':<12} {'Val_Loss':<12} {'MAE':<10} {'RMSE':<10} "
          f"{'R²':<10} {'Acc@3':<10} {'LR':<12} {'Status':<12}")
    print("-" * 118)

    stopped_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_out = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mean_t=mean_t,
            std_t=std_t,
            args=args,
            label_boundaries_t=label_boundaries_t,
            label_bin_weights=label_bin_weights,
            tail_threshold_y=tail_threshold_y,
        )
        train_loss_norm = float(train_out['loss_norm'])

        val_metrics = evaluate(model, val_loader, criterion, device, mean_t, std_t, args, postproc=None)
        val_loss_norm = float(val_metrics['loss_norm'])

        # Select the early-stopping metric.
        if args.early_stop_metric == 'mae':
            monitor = float(val_metrics['mae'])
        elif args.early_stop_metric == 'under_rate':
            monitor = float(val_metrics['under_rate'])
        else:
            monitor = float(val_metrics['mre'])


        if scheduler is not None:
            if mon_ema is None:
                mon_ema = monitor
            else:
                mon_ema = ema_alpha * monitor + (1.0 - ema_alpha) * float(mon_ema)
            scheduler.step(mon_ema)

        lr = float(optimizer.param_groups[0]['lr'])


        is_new_best = early_stop(epoch, monitor)

        status = f"Pat {early_stop.counter}/{args.patience}"
        if is_new_best:
            best_state = deepcopy(model.state_dict())
            best = {
                'epoch': int(epoch),
                'train_loss_norm': float(train_loss_norm),
                'val_loss_norm': float(val_loss_norm),
                **val_metrics
            }
            status = "✓ Best"


        history['epoch'].append(int(epoch))
        history['train_loss_norm'].append(float(train_loss_norm))
        history['val_loss_norm'].append(float(val_loss_norm))
        history['val_r2'].append(float(val_metrics['r2']))
        history['val_mae'].append(float(val_metrics['mae']))
        history['val_rmse'].append(float(val_metrics['rmse']))
        history['val_within_1'].append(float(val_metrics['within_1']))
        history['val_within_2'].append(float(val_metrics['within_2']))
        history['val_within_3'].append(float(val_metrics['within_3']))
        history['val_mre'].append(float(val_metrics['mre']))
        history['val_rel10'].append(float(val_metrics['rel10']))
        history['val_under_rate'].append(float(val_metrics['under_rate']))
        history['lr'].append(float(lr))
        history['monitor'].append(float(monitor))


        print(f"{epoch:<6} {train_loss_norm:<12.6f} {val_loss_norm:<12.6f} "
              f"{val_metrics['mae']:<10.4f} {val_metrics['rmse']:<10.4f} {val_metrics['r2']:<10.4f} "
              f"{val_metrics['within_3']:<10.4f} {lr:<12.2e} {status:<12}")

        if early_stop.early_stop:
            stopped_epoch = epoch

            print("\n" + "=" * 100)
            print(f"Early stopping triggered at epoch {epoch}; best epoch = {early_stop.best_epoch}")
            print("=" * 100)
            break


    pd.DataFrame(history).to_csv(os.path.join(save_dir, f'fold_{fold}_history.csv'), index=False)


    if best_state is not None:
        model.load_state_dict(best_state)


    postproc = fit_postproc_on_val(model, val_loader, device, mean_t, std_t, args)

    ckpt_path = os.path.join(save_dir, f'fold_{fold}_best.pth')
    torch.save({
        'fold': fold,
        'epoch': int(best.get('epoch', -1)),
        'model_state_dict': model.state_dict(),
        'best_monitor': float(early_stop.best_score),
        'best_val_metrics': best,
        'label_mean': float(mean),
        'label_std': float(std),
        'target_transform': args.target_transform,
        'postproc': postproc,
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
        'early_stopping_state': early_stop.state_dict(),
        'args_snapshot': vars(args),
    }, ckpt_path)


    print("\n" + "=" * 100)
    print(f"Fold {fold} training complete")
    print("=" * 100)
    print(f"Best epoch: {int(best.get('epoch', -1))}")
    print(f"Train_Loss: {float(best.get('train_loss_norm', float('nan'))):.6f} (norm)")
    print(f"Val_Loss:   {float(best.get('val_loss_norm', float('nan'))):.6f} (norm)")
    print(f"Val MAE:    {float(best.get('mae', float('nan'))):.4f}")
    print(f"Val RMSE:   {float(best.get('rmse', float('nan'))):.4f}")
    print(f"Acc@1(|err|<=1): {float(best.get('within_1', float('nan'))):.4f}")
    print(f"Acc@2(|err|<=2): {float(best.get('within_2', float('nan'))):.4f}")
    print(f"Acc@3(|err|<=3): {float(best.get('within_3', float('nan'))):.4f}")
    print(f"Val R²:     {float(best.get('r2', float('nan'))):.4f}")
    print(f"Model saved: {os.path.basename(ckpt_path)}")
    print("=" * 100)

    return {
        'fold': fold,
        'best_epoch': int(best.get('epoch', -1)),
        'train_loss_norm': float(best.get('train_loss_norm', float('nan'))),
        'val_loss_norm': float(best.get('val_loss_norm', float('nan'))),
        'val_r2': float(best.get('r2', float('nan'))),
        'val_mae': float(best.get('mae', float('nan'))),
        'val_rmse': float(best.get('rmse', float('nan'))),
        'val_within_1': float(best.get('within_1', float('nan'))),
        'val_within_2': float(best.get('within_2', float('nan'))),
        'val_within_3': float(best.get('within_3', float('nan'))),
        'val_mre': float(best.get('mre', float('nan'))),
        'val_rel10': float(best.get('rel10', float('nan'))),
        'val_under_rate': float(best.get('under_rate', float('nan'))),
        'post_offset': float(postproc.get('offset', 0.0)),
    }


# Main workflow
def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{args.model_type}_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("Starting training with label-only stratification and an underestimation penalty")
    print("=" * 70)
    print(f"Output directory: {save_dir}")
    print(f"Device: {device}")

    graphs = torch.load(args.data_path, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]
    if len(graphs) == 0:
        raise ValueError("The dataset is empty")
    print(f"\nLoaded {len(graphs)} LCG graphs (HeteroData)")

    s = graphs[0]
    if not hasattr(s, 'struct_x'):
        raise ValueError("The dataset is missing struct_x; verify that the input contains trimmed *_with_features.pt training graphs")

    sx = s.struct_x
    if not torch.is_tensor(sx):
        raise ValueError(f"struct_x must be a Tensor, got {type(sx)}")

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
    print(f"Feature dimensions from LCG data: pos={feature_dims['pos_dim']}, "
          f"neg={feature_dims['neg_dim']}, clause={feature_dims['clause_dim']}, struct={feature_dims['struct_dim']}")

    ys = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
    print(f"\nLabel statistics: range [{ys.min():.2f}, {ys.max():.2f}], "
          f"mean {ys.mean():.2f}, standard deviation {ys.std():.2f}, median {np.median(ys):.2f}")

    results = []
    print("\n" + "=" * 70)
    print(f"Starting {args.n_splits} y-stratified splits and training runs")
    print("=" * 70)

    for fold in range(1, args.n_splits + 1):
        print("\n" + "=" * 69)
        print(f"Fold {fold}/{args.n_splits}")
        print("=" * 69)

        split_seed = args.seed + fold - 1
        train_idx, val_idx, test_idx = split_train_val_test_stratified_by_y(
            graphs,
            seed=split_seed,
            label_bins=args.label_bins,
            target_transform=args.target_transform,
        )

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        print(f"Training set: {len(train_graphs)} graphs, validation set: {len(val_graphs)} graphs, test set: {len(test_graphs)} graphs")

        mean, std = compute_normalization_params(train_graphs, args.target_transform)
        print(f"Label normalization in {args.target_transform} space: mean={mean:.4f}, std={std:.4f}")

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

        if fold == 1:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameter count: {params:,}")

        fold_result = train_fold(
            fold, model,
            train_graphs, val_graphs,
            args, device, mean, std,
            save_dir,
            feature_dims=feature_dims
        )


        ckpt_path = os.path.join(save_dir, f'fold_{fold}_best.pth')
        ckpt = torch.load(ckpt_path, map_location=device)

        mc = ckpt.get('model_config', {})
        test_model = build_model(
            model_type=mc.get('model_type', args.model_type),
            pos_in_dim=mc.get('pos_in_dim', feature_dims['pos_dim']),
            neg_in_dim=mc.get('neg_in_dim', feature_dims['neg_dim']),
            clause_in_dim=mc.get('clause_in_dim', feature_dims['clause_dim']),
            hidden_dim=mc.get('hidden_dim', 128),
            num_layers=mc.get('num_layers', 4),
            dropout=mc.get('dropout', 0.1),
            attn_pool_dropout=mc.get('attn_pool_dropout', 0.1),
            attn_lambda_init=mc.get('attn_lambda_init', 0.1),
            use_struct=mc.get('use_struct', True),
            struct_in_dim=mc.get('struct_in_dim', feature_dims['struct_dim']),
            gat_heads=mc.get('gat_heads', 4),
            strict_input_dim_check=mc.get('strict_input_dim_check', True),
        ).to(device)
        test_model.load_state_dict(ckpt['model_state_dict'])

        mean_t = torch.tensor(float(ckpt['label_mean']), dtype=torch.float32, device=device)
        std_t = torch.tensor(float(ckpt['label_std']), dtype=torch.float32, device=device)


        args_test = deepcopy(args)
        args_test.target_transform = ckpt.get('target_transform', args.target_transform)
        postproc = ckpt.get('postproc', None)

        test_metrics = evaluate(
            test_model,
            DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
            nn.HuberLoss(delta=args.huber_delta, reduction='none'),
            device, mean_t, std_t, args_test,
            postproc=postproc
        )

        offset_y = 0.0
        if postproc is not None:
            offset_y = float(postproc.get('offset', 0.0))


        print(f"[Fold {fold}] Test results (offset_y={offset_y:.6f})")
        print(f"  Test MAE :  {test_metrics['mae']:.4f}")
        print(f"  Test RMSE:  {test_metrics['rmse']:.4f}")
        print(f"  Test R²  :  {test_metrics['r2']:.4f}")
        print(f"  Test Acc@1(|err|<=1): {test_metrics['within_1']:.4f}")
        print(f"  Test Acc@2(|err|<=2): {test_metrics['within_2']:.4f}")
        print(f"  Test Acc@3(|err|<=3): {test_metrics['within_3']:.4f}")

        fold_result.update({
            'test_r2': float(test_metrics['r2']),
            'test_mae': float(test_metrics['mae']),
            'test_rmse': float(test_metrics['rmse']),
            'test_within_1': float(test_metrics['within_1']),
            'test_within_2': float(test_metrics['within_2']),
            'test_within_3': float(test_metrics['within_3']),
            'test_mre': float(test_metrics['mre']),
            'test_rel10': float(test_metrics['rel10']),
            'test_under_rate': float(test_metrics['under_rate']),
        })
        results.append(fold_result)


    print("\n" + "=" * 100)
    print(f"Completed {args.n_splits} label-only stratified training runs")
    print("=" * 100)

    df = pd.DataFrame(results)

    cols = [
        'fold', 'best_epoch',
        'val_r2', 'val_mae', 'val_rmse', 'val_within_1', 'val_within_2', 'val_within_3',
        'test_r2', 'test_mae', 'test_rmse', 'test_within_1', 'test_within_2', 'test_within_3'
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    print("\nDetailed results by fold:")
    float_cols = [c for c in cols if c not in ('fold', 'best_epoch')]
    formatters = {c: (lambda x: f"{float(x):.6f}") for c in float_cols}
    print(df[cols].to_string(index=False, formatters=formatters))

    def _mean_std(col: str):
        v = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 0.0
        return float(v.mean()), float(v.std(ddof=0))

    m, s = _mean_std('val_mae')
    print(f"\nMean validation MAE: {m:.4f} ± {s:.4f}")
    m, s = _mean_std('test_mae')
    print(f"Mean test MAE: {m:.4f} ± {s:.4f}")

    csv_path = os.path.join(save_dir, "all_folds_results.csv")
    df[cols].to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print(f"Results and models saved to: {save_dir}")
    print("=" * 100)
    print(f"\nDetailed results and summary written to: {csv_path}\n")


if __name__ == '__main__':
    main()
