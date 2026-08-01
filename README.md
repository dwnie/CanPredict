# CanPredict

CanPredict estimates the size of a constrained covering array directly from a
feature-enriched literal–clause graph (LCG).  It supports 2-way and 3-way
interaction strengths.  The main predictor combines heterogeneous message
passing over positive-literal, negative-literal, and clause nodes with a
graph-level structural descriptor.

The repository also includes the APPTS and PI-APPTS implementation, processed
experimental records, and the artifacts needed to reproduce the reported
baselines and ablations.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/prediction/` | Heterogeneous LCG model definition, training, and target inference scripts. |
| `scripts/ablation/` | Ablation scripts: descriptor-only Ridge/XGBoost baselines and the LCG-without-structural-descriptor HeteroSAGE variant. |
| `src/APPTS_and_PI-APPTS/` | APPTS and PI-APPTS source code and benchmark input files. |
| `corpus/` | Serialized 2-way and 3-way LCG training corpora. Distributed as external research artifacts; see [Large artifacts](#large-artifacts). |
| `target-benchmarks/` | The 55 labeled target benchmark LCGs used for prediction evaluation. |
| `training-model/` | Ten-fold checkpoints for GNN models and archived Ridge/XGBoost model artifacts. Distributed as external research artifacts. |
| `data/` | Published predictions and aggregate tables for the generator experiments. |

## Requirements

- Python 3.10 or later
- PyTorch and PyTorch Geometric compatible with the selected CPU or CUDA
  environment
- Java 11 or later for APPTS and PI-APPTS

Install the non-PyTorch Python packages with:

```bash
python -m pip install numpy pandas scikit-learn joblib xgboost openpyxl
```

Install PyTorch and PyTorch Geometric using the platform-specific instructions
from their official documentation.

## Training the main LCG predictor

Run commands from the repository root.  The following examples train a
HeteroSAGE predictor and create ten fold-specific checkpoints.

```bash
python scripts/prediction/train_hetero_lcg_2way.py \
  --data_path corpus/t2_corpus.pt \
  --output_dir runs/2-way/heterosage \
  --model_type hetero_sage \
  --device cuda:0

python scripts/prediction/train_hetero_lcg_3way.py \
  --data_path corpus/t3_corpus.pt \
  --output_dir runs/3-way/heterosage \
  --model_type hetero_sage \
  --device cuda:0
```

Use `--device cpu` when CUDA is unavailable.  `hetero_gat` and `hetero_gin`
are also accepted model types.

## Predicting the 55 target benchmarks

The inference scripts ensemble the ten checkpoints in `--model_dir`.

```bash
python scripts/prediction/predict_hetero_lcg_2way.py \
  --data_path target-benchmarks/t2_55_benchmarks.pt \
  --model_type hetero_sage \
  --model_dir training-model/2-way/HeteroSAGE \
  --output_dir runs/predictions/2-way \
  --device cuda:0

python scripts/prediction/predict_hetero_lcg_3way.py \
  --data_path target-benchmarks/t3_55_benchmarks.pt \
  --model_type hetero_sage \
  --model_dir training-model/3-way/HeteroSAGE \
  --output_dir runs/predictions/3-way \
  --device cuda:0
```

The output directory contains the ensemble predictions and the corresponding
evaluation metrics.  The supplied target files include labels and therefore
support evaluation as well as inference.

## Ablation studies

The ablation study contains three model families.

1. **LCG-without-structural-descriptor HeteroSAGE** removes the graph-level
   structural descriptor while retaining the typed LCG message-passing path.
   Train it with `--use_struct false`:

   ```bash
   python scripts/ablation/train_lcg_without_structure_2way.py \
     --data_path corpus/t2_corpus.pt \
     --output_dir runs/2-way/lcg-without-struct \
     --model_type hetero_sage \
     --use_struct false \
     --device cuda:0

   python scripts/ablation/train_lcg_without_structure_3way.py \
     --data_path corpus/t3_corpus.pt \
     --output_dir runs/3-way/lcg-without-struct \
     --model_type hetero_sage \
     --use_struct false \
     --device cuda:0
   ```

2. **Ridge regression** is a descriptor-only linear baseline.
3. **XGBoost** is a descriptor-only nonlinear baseline.

The descriptor baseline inputs are in `scripts/ablation/data/`.  The
reproduction runner retrains both Ridge and XGBoost, saves the ten fold models,
scalers, splits, predictions, metrics, environment information, and compares
the result with the archived tables:

```bash
python scripts/ablation/reproduce_descriptor_baselines.py \
  --data-dir scripts/ablation/data \
  --reference-dir data/descriptor-baselines/results \
  --artifacts-dir runs/descriptor-baselines \
  --ridge-alpha 1.0
```

The archived model artifacts are under `training-model/{2-way,3-way}/Ridge`,
`training-model/{2-way,3-way}/XGBoost`, and the current
`training-model/{2-way,3-way}/lcg-without-structural-descriptors-heterosage`.

## APPTS and PI-APPTS

The dedicated [APPTS and PI-APPTS README](src/APPTS_and_PI-APPTS/README.md)
documents the Java command-line interface and the optional predicted initial
size used by PI-APPTS.

## Large artifacts

The two serialized corpora and the archived model artifacts are available in
the [CanPredict Zenodo record](https://doi.org/10.5281/zenodo.21740539),
version `v1.0.0`.  The record includes SHA-256 checksums and restoration
instructions.

The public Git repository should include the code, documentation, small target
benchmark files, processed tables, and raw JSON records.  Users should download
the external corpus and checkpoint artifacts and place them at:

```text
corpus/t2_corpus.pt
corpus/t3_corpus.pt
training-model/
```

Git LFS is suitable only for artifacts that are within the applicable LFS
per-file limit and storage quota.  It is not a good default for the two large
corpus files.

## Naming convention

Public-facing scripts and artifact directories use portable lowercase names.
The 2-way and 3-way suffixes are written as `2way` and `3way`; directories use
hyphens to separate words.  This convention avoids operating-system-specific
characters and makes commands stable across platforms.
