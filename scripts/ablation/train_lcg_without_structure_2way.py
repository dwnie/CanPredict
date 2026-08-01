#!/usr/bin/env python3
"""
train_lcg_without_structure_2way.py — LCG-without-structure ablation for
2-way HeteroSAGE training.



Usage:
  python train_lcg_without_structure_2way.py \\
      --data_path .../t2_corpus.pt \\
      --output_dir ./ablation_t2_nostruct \\
      --use_struct False

"""

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

# Import the ablation model from this directory.
import sys
_ablation_dir = os.path.dirname(os.path.abspath(__file__))
if _ablation_dir not in sys.path:
    sys.path.insert(0, _ablation_dir)
from hetero_lcg_model import build_model


# ======================================================================
#  [CHANGE 1] Added --use_struct argument
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Ablation: HeteroSAGE w/ or w/o struct_x (2-way)')

    # Data & output
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='saved_models_ablation_t2')

    # Model
    p.add_argument('--model_type', type=str, default='hetero_sage',
                   choices=['hetero_sage', 'hetero_gin', 'hetero_gat'])
    p.add_argument('--hidden_dim', type=int, default=256)  # matches original config
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--gat_heads', type=int, default=4)
    p.add_argument('--attn_pool_dropout', type=float, default=0.1)
    p.add_argument('--attn_lambda_init', type=float, default=0.1)

    # [CHANGE 1a] New argument to toggle struct_x
    p.add_argument('--use_struct', type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                   default=True,
                   help='Whether to include graph-level struct_x features (default: True)')

    # Training
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=64)   # matches original config
    p.add_argument('--weight_decay', type=float, default=1e-5)

    # Label transform
    p.add_argument('--target_transform', type=str, default='log1p', choices=['none', 'log1p'])

    # Loss
    p.add_argument('--loss_type', type=str, default='rel_asym_huber',
                   choices=['huber_norm', 'rel_asym_huber'])
    p.add_argument('--huber_delta', type=float, default=1.0)
    p.add_argument('--rel_huber_delta', type=float, default=0.10)
    p.add_argument('--rel_denominator_floor', type=float, default=1e-6)
    p.add_argument('--under_weight', type=float, default=2.5)
    p.add_argument('--over_weight', type=float, default=1.0)
    p.add_argument('--y_weight_power', type=float, default=0.25)
    p.add_argument('--y_weight_clip', type=float, default=5.0)

    # Cross-validation
    p.add_argument('--n_splits', type=int, default=10)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--use_scheduler', action='store_true', default=False)
    p.add_argument('--scheduler_patience', type=int, default=8)

    p.add_argument('--early_stop_metric', type=str, default='mre',
                   choices=['mae', 'mre', 'under_rate'])

    # Label bins
    p.add_argument('--label_bins', type=int, default=8)
    p.add_argument('--no_label_weight', action='store_true', default=False)
    p.add_argument('--label_weight_power', type=float, default=1.0)
    p.add_argument('--tail_quantile', type=float, default=0.90)
    p.add_argument('--tail_min_y', type=float, default=50.0)
    p.add_argument('--tail_boost', type=float, default=2.0)
    p.add_argument('--small_y', type=float, default=20.0)
    p.add_argument('--small_boost', type=float, default=1.3)

    # Post-processing
    p.add_argument('--enable_calibration', action='store_true', default=True)
    p.add_argument('--calibration_clip_a', type=float, default=2.0)
    p.add_argument('--calibration_clip_b', type=float, default=2.0)
    p.add_argument('--enable_one_sided_offset', action='store_true', default=True)
    p.add_argument('--offset_quantile', type=float, default=0.90)
    p.add_argument('--offset_max', type=float, default=50.0)

    # Other
    p.add_argument('--device', type=str, default='cuda:0',
                   choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_interval', type=int, default=10)

    return p.parse_args()


# ======================================================================
#  The remaining code follows the main 2-way training protocol, with the
#  structural descriptor disabled during model construction.
# ======================================================================

# ── Label transform ─────────────────────────────────────────────────────

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


# ── Normalization ───────────────────────────────────────────────────────

def compute_normalization_params(train_graphs, target_transform='none'):
    labels = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    labels_t = transform_y_np(labels, target_transform)
    mean = float(labels_t.mean()) if labels_t.size else 0.0
    std = float(labels_t.std()) if labels_t.size else 1.0
    if std < 1e-8:
        std = 1.0
    return mean, std


# ── Label bins ──────────────────────────────────────────────────────────

def _build_label_bins_boundaries_np(labels, label_bins, target_transform):
    if label_bins is None or int(label_bins) <= 1:
        return None
    labels_t = transform_y_np(labels.astype(np.float32), target_transform)
    qs = np.quantile(labels_t, np.linspace(0.0, 1.0, int(label_bins) + 1))
    qs = np.unique(qs)
    if qs.size <= 2:
        return None
    return qs[1:-1].astype(np.float32)


def _assign_bin_ids_np(labels, boundaries_t, target_transform):
    if boundaries_t is None or boundaries_t.size == 0:
        return np.zeros((labels.shape[0],), dtype=np.int64)
    labels_t = transform_y_np(labels.astype(np.float32), target_transform)
    return np.digitize(labels_t, boundaries_t, right=True).astype(np.int64)


def compute_label_weighting(train_graphs, label_bins, target_transform,
                             label_weight_power=1.0, tail_quantile=0.90,
                             tail_min_y=50.0):
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


# ── Batch sampler ───────────────────────────────────────────────────────

class LabelStratifiedBatchSampler(Sampler):
    def __init__(self, graphs, batch_size, seed=42, label_bins=8, target_transform='log1p'):
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
        self.bin2idx = {}
        for i, b in enumerate(bin_ids.tolist()):
            self.bin2idx.setdefault(int(b), []).append(i)
        self._reset_iters()

    def _reset_iters(self):
        self.bin_iters = {}
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


# ── Early stopping ──────────────────────────────────────────────────────

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


# ── Stratified split ────────────────────────────────────────────────────

def split_train_val_test_stratified_by_y(graphs, seed=42, label_bins=8,
                                          target_transform='log1p'):
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


# ── Metrics / post-processing ───────────────────────────────────────────

def _attach_batch_size(batch):
    if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
        bs = int(batch.num_graphs)
    else:
        bs = int(batch.y.view(-1).numel())
    batch.batch_size = bs
    return bs


def _relative_huber(rel_err_signed, delta):
    abs_e = torch.abs(rel_err_signed)
    quad = 0.5 * (abs_e ** 2)
    lin = delta * (abs_e - 0.5 * delta)
    return torch.where(abs_e <= delta, quad, lin)


def compute_rel_metrics(labels, preds, floor=1e-6):
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.size == 0:
        return {'mre': 0.0, 'rel10': 0.0, 'under_rate': 0.0}
    denom = np.maximum(labels, float(floor))
    rel = np.abs(preds - labels) / denom
    rel = np.nan_to_num(rel, nan=0.0, posinf=1e9, neginf=0.0)
    under = (preds < labels).astype(np.float32)
    return {'mre': float(rel.mean()), 'rel10': float((rel <= 0.10).mean()),
            'under_rate': float(under.mean())}


def fit_affine_calibration_in_t(pred_t_np, y_t_np, clip_a=2.0, clip_b=2.0):
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


def apply_postprocess(pred_t_raw, target_transform, calib=None, offset=0.0):
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
def collect_val_pred_t_and_y_t(model, loader, device, mean_t, std_t, target_transform):
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


def fit_postproc_on_val(model, val_loader, device, mean_t, std_t, args):
    postproc = {'calib': None, 'offset': 0.0}
    pred_t_raw_np, y_t_np = collect_val_pred_t_and_y_t(
        model, val_loader, device, mean_t, std_t, args.target_transform)
    if pred_t_raw_np.size == 0:
        return postproc
    calib = None
    if bool(args.enable_calibration):
        calib = fit_affine_calibration_in_t(
            pred_t_raw_np, y_t_np,
            clip_a=float(args.calibration_clip_a),
            clip_b=float(args.calibration_clip_b))
    pred_t_raw_t = torch.from_numpy(pred_t_raw_np).to(torch.float32)
    pred_cal = apply_postprocess(pred_t_raw_t, args.target_transform,
                                  calib=calib, offset=0.0).cpu().numpy().reshape(-1)
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


# ── Train / eval ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device,
                mean_t, std_t, args,
                label_boundaries_t=None, label_bin_weights=None,
                tail_threshold_y=None):
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
        with torch.no_grad():
            y_t = transform_y_tensor(y, args.target_transform)
            y_norm = (y_t - mean_t) / std_t
            loss_norm_vec = criterion(pred_norm, y_norm)
            total_loss_norm += float(loss_norm_vec.sum().item())
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
    return {'loss_main': (total_loss_main / total if total > 0 else 0.0),
            'loss_norm': (total_loss_norm / total if total > 0 else 0.0)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, mean_t, std_t, args,
             postproc=None):
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
        y_t = transform_y_tensor(y, args.target_transform)
        y_norm = (y_t - mean_t) / std_t
        loss_vec_norm = criterion(pred_norm, y_norm)
        total_loss_norm += float(loss_vec_norm.sum().item())
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
    return {'loss': float(avg_loss_norm), 'loss_norm': float(avg_loss_norm),
            'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse),
            'within_1': float(within_1), 'within_2': float(within_2),
            'within_3': float(within_3), 'mre': float(relm['mre']),
            'rel10': float(relm['rel10']), 'under_rate': float(relm['under_rate'])}


# ── Single fold training ────────────────────────────────────────────────

def train_fold(fold, model, train_graphs, val_graphs, args, device,
               mean, std, save_dir, feature_dims):
    label_boundaries_t, label_bin_weights, tail_threshold_y = compute_label_weighting(
        train_graphs=train_graphs, label_bins=args.label_bins,
        target_transform=args.target_transform,
        label_weight_power=args.label_weight_power,
        tail_quantile=args.tail_quantile, tail_min_y=args.tail_min_y)
    criterion = nn.HuberLoss(delta=args.huber_delta, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-6)
    early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta,
                                warmup=args.warmup_epochs, verbose=False)
    mean_t = torch.tensor(float(mean), dtype=torch.float32, device=device)
    std_t = torch.tensor(float(std), dtype=torch.float32, device=device)
    train_sampler = LabelStratifiedBatchSampler(
        train_graphs, batch_size=args.batch_size, seed=args.seed + fold,
        label_bins=args.label_bins, target_transform=args.target_transform)
    train_loader = DataLoader(train_graphs, batch_sampler=train_sampler,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    best = {}
    best_state = None
    history = {'epoch': [], 'train_loss_norm': [], 'val_loss_norm': [],
               'val_r2': [], 'val_mae': [], 'val_rmse': [],
               'val_within_1': [], 'val_within_2': [], 'val_within_3': [],
               'val_mre': [], 'val_rel10': [], 'val_under_rate': [], 'lr': [],
               'monitor': []}

    print("\n" + "-" * 118)
    print(f"Fold {fold} training (use_struct={args.use_struct})")
    print("-" * 118)

    for epoch in range(1, args.epochs + 1):
        train_out = train_epoch(model=model, loader=train_loader, criterion=criterion,
                                optimizer=optimizer, device=device, mean_t=mean_t,
                                std_t=std_t, args=args,
                                label_boundaries_t=label_boundaries_t,
                                label_bin_weights=label_bin_weights,
                                tail_threshold_y=tail_threshold_y)
        train_loss_norm = float(train_out['loss_norm'])
        val_metrics = evaluate(model, val_loader, criterion, device, mean_t, std_t,
                               args, postproc=None)
        val_loss_norm = float(val_metrics['loss_norm'])
        if args.early_stop_metric == 'mae':
            monitor = float(val_metrics['mae'])
        elif args.early_stop_metric == 'under_rate':
            monitor = float(val_metrics['under_rate'])
        else:
            monitor = float(val_metrics['mre'])
        if scheduler is not None:
            scheduler.step(monitor)
        lr = float(optimizer.param_groups[0]['lr'])
        is_new_best = early_stop(epoch, monitor)
        status = f"Pat {early_stop.counter}/{args.patience}"
        if is_new_best:
            best_state = deepcopy(model.state_dict())
            best = {'epoch': int(epoch), 'train_loss_norm': float(train_loss_norm),
                    'val_loss_norm': float(val_loss_norm), **val_metrics}
            status = "✓ Best"
        history['epoch'].append(int(epoch))
        history['train_loss_norm'].append(float(train_loss_norm))
        history['val_loss_norm'].append(float(val_loss_norm))
        for k in ['r2', 'mae', 'rmse', 'within_1', 'within_2', 'within_3',
                   'mre', 'rel10', 'under_rate']:
            history[f'val_{k}'].append(float(val_metrics[k]))
        history['lr'].append(float(lr))
        history['monitor'].append(float(monitor))
        print(f"{epoch:<6} {train_loss_norm:<12.6f} {val_loss_norm:<12.6f} "
              f"{val_metrics['mae']:<10.4f} {val_metrics['rmse']:<10.4f} {val_metrics['r2']:<10.4f} "
              f"{val_metrics['within_3']:<10.4f} {lr:<12.2e} {status:<12}")
        if early_stop.early_stop:
            print("\n" + "=" * 100)
            print(f"Early stopping at epoch {epoch} (best = {early_stop.best_epoch})")
            print("=" * 100)
            break

    pd.DataFrame(history).to_csv(os.path.join(save_dir, f'fold_{fold}_history.csv'), index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    postproc = fit_postproc_on_val(model, val_loader, device, mean_t, std_t, args)

    # [CHANGE 3] Save use_struct in model_config
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
            'use_struct': bool(args.use_struct),               # ← saved
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
    print(f"Fold {fold} complete (use_struct={args.use_struct})")
    print("=" * 100)
    print(f"Best epoch: {int(best.get('epoch', -1))}")
    print(f"Val R²:     {float(best.get('r2', float('nan'))):.4f}")
    print(f"Val MAE:    {float(best.get('mae', float('nan'))):.4f}")
    print(f"Val RMSE:   {float(best.get('rmse', float('nan'))):.4f}")
    print(f"Model saved: {os.path.basename(ckpt_path)}")

    return {'fold': fold, 'best_epoch': int(best.get('epoch', -1)),
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
            'post_offset': float(postproc.get('offset', 0.0))}


# ── Main ────────────────────────────────────────────────────────────────

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
    print(f"Ablation Training (use_struct={args.use_struct})")
    print("=" * 70)
    print(f"Save dir: {save_dir}")
    print(f"Device:   {device}")

    graphs = torch.load(args.data_path, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]
    if len(graphs) == 0:
        raise ValueError("Empty dataset")
    print(f"\nLoaded {len(graphs)} LCG graphs")

    s = graphs[0]
    if not hasattr(s, 'struct_x'):
        raise ValueError("Data missing struct_x")
    sx = s.struct_x
    if sx.dim() == 2:
        struct_dim = int(sx.shape[1])
    elif sx.dim() == 1:
        struct_dim = int(sx.shape[0])
    elif sx.dim() == 0:
        struct_dim = 1
    else:
        raise ValueError(f"struct_x shape anomaly: {tuple(sx.shape)}")

    feature_dims = {
        'pos_dim': int(s['pos'].x.shape[1]),
        'neg_dim': int(s['neg'].x.shape[1]),
        'clause_dim': int(s['clause'].x.shape[1]),
        'struct_dim': int(struct_dim),
    }
    print(f"Feature dims: {feature_dims}")

    ys = np.array([float(g.y.item()) for g in graphs], dtype=np.float32)
    print(f"Labels: [{ys.min():.2f}, {ys.max():.2f}], mean={ys.mean():.2f}")

    results = []
    for fold in range(1, args.n_splits + 1):
        split_seed = args.seed + fold - 1
        train_idx, val_idx, test_idx = split_train_val_test_stratified_by_y(
            graphs, seed=split_seed, label_bins=args.label_bins,
            target_transform=args.target_transform)
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]
        print(f"\nFold {fold}/{args.n_splits}: train={len(train_graphs)} "
              f"val={len(val_graphs)} test={len(test_graphs)}")
        mean, std = compute_normalization_params(train_graphs, args.target_transform)
        print(f"Normalization: mean={mean:.4f}, std={std:.4f}")

        # [CHANGE 2] Pass use_struct to build_model
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
            use_struct=bool(args.use_struct),             # ← from command line
            struct_in_dim=feature_dims['struct_dim'],
            gat_heads=args.gat_heads,
            strict_input_dim_check=True,
        ).to(device)

        if fold == 1:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,}")

        fold_result = train_fold(fold, model, train_graphs, val_graphs, args,
                                 device, mean, std, save_dir, feature_dims=feature_dims)

        # Test evaluation
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
            use_struct=bool(mc.get('use_struct', True)),  # ← from checkpoint
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
            DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers),
            nn.HuberLoss(delta=args.huber_delta, reduction='none'),
            device, mean_t, std_t, args_test, postproc=postproc)
        offset_y = 0.0
        if postproc is not None:
            offset_y = float(postproc.get('offset', 0.0))
        print(f"[Fold {fold}] Test (offset_y={offset_y:.6f}): "
              f"MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} "
              f"R²={test_metrics['r2']:.4f}")
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

    df = pd.DataFrame(results)
    cols = ['fold', 'best_epoch', 'val_r2', 'val_mae', 'val_rmse',
            'val_within_1', 'val_within_2', 'val_within_3',
            'test_r2', 'test_mae', 'test_rmse', 'test_within_1',
            'test_within_2', 'test_within_3']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    print("\nFold results:")
    print(df[cols].to_string(index=False))
    csv_path = os.path.join(save_dir, "all_folds_results.csv")
    df[cols].to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Model dir: {save_dir}")


if __name__ == '__main__':
    main()
