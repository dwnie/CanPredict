#!/usr/bin/env python3


import os
import sys
import argparse
import csv
import warnings

import numpy as np
import torch


# ======================================================================
#  Struct-x feature names (15-d, order matches feature_engineering.py)
# ======================================================================
STRUCT_X_FEATURE_NAMES = [
    "log1p_P",              # 0
    "log1p_M",              # 1
    "mean_domain_size",     # 2
    "domain_entropy",       # 3
    "log1p_num_clauses",    # 4
    "ratio_FT",             # 5
    "ratio_AMO",            # 6
    "ratio_unit",           # 7
    "ratio_cross_FT",       # 8
    "FT_len_mean",          # 9
    "FT_len_max",           # 10
    "clause_len_mean",      # 11
    "density_neg2clause",   # 12
    "max_degn2c_norm",      # 13
    "gini_degn2c",          # 14
]

EXPECTED_STRUCT_DIM = 15


# ======================================================================
#  Data extraction helpers
# ======================================================================

def _safe_extract_struct_x(graph, idx: int) -> np.ndarray:
    """Extract struct_x as a 1-D float64 numpy array of length 15.

    Supports struct_x with shape [1, D], [D], or scalar.
    Raises ValueError if shape is unexpected.
    """
    if not hasattr(graph, "struct_x"):
        raise ValueError(f"graph[{idx}] has no 'struct_x' attribute")

    sx = graph.struct_x
    if not torch.is_tensor(sx):
        raise TypeError(f"graph[{idx}].struct_x is not a Tensor (got {type(sx)})")

    # Flatten to 1-D
    arr = sx.detach().cpu().numpy().flatten().astype(np.float64)

    if arr.size != EXPECTED_STRUCT_DIM:
        raise ValueError(
            f"graph[{idx}].struct_x has {arr.size} elements, "
            f"expected {EXPECTED_STRUCT_DIM}. Aborting — do not proceed with "
            f"mismatched descriptor dimensions."
        )

    return arr


def _safe_extract_label(graph, idx: int) -> float:
    """Extract the regression label y (covering array size) as a float."""
    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError(f"graph[{idx}] has no 'y' attribute")
    if torch.is_tensor(y):
        return float(y.detach().cpu().item())
    return float(y)


def _load_graphs(pt_path: str, label: str) -> list:
    """Load .pt file and validate it is a non-empty list of HeteroData objects."""
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"{label} data not found: {pt_path}")

    print(f"[extract] Loading {label} data: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, list):
        graphs = obj
    else:
        # Some pipelines save a Dataset or single graph; try to convert
        if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            graphs = [obj[i] for i in range(len(obj))]
        else:
            graphs = [obj]

    if len(graphs) == 0:
        raise ValueError(f"{label} data is empty ({pt_path})")

    print(f"[extract]   Loaded {len(graphs)} graphs")
    return graphs


# ======================================================================
#  CSV writer
# ======================================================================

def _write_csv(csv_path: str, rows: list, fieldnames: list):
    """Write a list of dicts to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[extract]   Wrote {len(rows)} rows to {csv_path}")


def _build_data_rows(graphs: list) -> tuple:
    """Extract (label, struct_x) from every graph in the list.

    Returns (rows, fieldnames) where rows is a list of dicts suitable for
    CSV DictWriter.
    """
    rows = []
    feat_cols = [f"feat_{i}" for i in range(EXPECTED_STRUCT_DIM)]

    for idx, g in enumerate(graphs):
        label = _safe_extract_label(g, idx)
        sx = _safe_extract_struct_x(g, idx)

        row = {"benchmark_id": idx, "label": label}
        for i in range(EXPECTED_STRUCT_DIM):
            row[f"feat_{i}"] = float(sx[i])
        rows.append(row)

    fieldnames = ["benchmark_id", "label"] + feat_cols
    return rows, fieldnames


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract graph-level structural descriptors from LCG .pt files"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to write CSV files (default: ./output)"
    )

    # Training data paths
    _base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _data_dir = os.path.join(
        _base,
        "datasets"
    )

    parser.add_argument(
        "--t2_data_path", type=str,
        default=os.path.join(
            _data_dir,
            "t2_new_dataset_clean_lcg_1231",
            "t2_new_data_lcgs_with_features1231.pt"
        ),
        help="2-way training corpus .pt path"
    )
    parser.add_argument(
        "--t3_data_path", type=str,
        default=os.path.join(
            _data_dir,
            "t3_new_dataset_clean_lcg_1231",
            "t3_new_data_lcgs_with_features1231.pt"
        ),
        help="3-way training corpus .pt path"
    )

    # Target benchmark paths
    parser.add_argument(
        "--t2_target_path", type=str,
        default=os.path.join(
            _data_dir,
            "groundTruthModel", "model_t2_sut", "model_t2_lcg",
            "model_t2_lcgs_with_features12311.pt"
        ),
        help="2-way target benchmark .pt path"
    )
    parser.add_argument(
        "--t3_target_path", type=str,
        default=os.path.join(
            _data_dir,
            "groundTruthModel", "model_t3_sut", "model_t3_lcg",
            "model_t3_lcgs_with_features1231.pt"
        ),
        help="3-way target benchmark .pt path"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Graph Descriptor Extraction")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Expected struct_x dimension: {EXPECTED_STRUCT_DIM}")
    print()

    # ── Extract each dataset ──────────────────────────────────────────
    datasets = [
        ("t2_data",   args.t2_data_path,   "descriptor_baseline_t2_data.csv"),
        ("t3_data",   args.t3_data_path,   "descriptor_baseline_t3_data.csv"),
        ("t2_target", args.t2_target_path, "descriptor_baseline_t2_target.csv"),
        ("t3_target", args.t3_target_path, "descriptor_baseline_t3_target.csv"),
    ]

    for dataset_name, pt_path, csv_name in datasets:
        print(f"\n[{dataset_name}]")
        try:
            graphs = _load_graphs(pt_path, dataset_name)
            rows, fieldnames = _build_data_rows(graphs)

            # Validate first graph's struct_x
            _safe_extract_struct_x(graphs[0], 0)
            print(f"[extract]   struct_x dimension: {EXPECTED_STRUCT_DIM} ✓")

            csv_path = os.path.join(args.output_dir, csv_name)
            _write_csv(csv_path, rows, fieldnames)

        except (FileNotFoundError, ValueError, TypeError) as e:
            print(f"[extract]   ERROR: {e}", file=sys.stderr)
            print(f"[extract]   SKIPPING {dataset_name}", file=sys.stderr)
            continue

    # ── Write feature list ────────────────────────────────────────────
    feat_list_path = os.path.join(args.output_dir, "descriptor_feature_list.txt")
    with open(feat_list_path, "w", encoding="utf-8") as f:
        f.write(f"Graph-level structural descriptors ({EXPECTED_STRUCT_DIM} features)\n")
        f.write(f"Source: OptimalFeatureCalculator._calculate_graph_level_struct_x()\n")
        f.write(f"Extracted from: data['struct_x'] in trimmed training graphs\n")
        f.write(f"{'=' * 50}\n")
        for i, name in enumerate(STRUCT_X_FEATURE_NAMES):
            f.write(f"  {i:2d}. {name}\n")

    print(f"\n[extract] Feature list written to {feat_list_path}")
    print("[extract] Done.")


if __name__ == "__main__":
    main()
