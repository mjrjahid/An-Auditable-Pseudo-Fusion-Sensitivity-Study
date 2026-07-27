# An-Auditable-Pseudo-Fusion-Sensitivity-Study

A standalone Python script that builds a **sample-level fused dataset** from disk, memory,
and network telemetry, and rigorously benchmarks classifiers on it — with an explicit
guardrail against silently degrading into a fake ("pseudo") fusion when true sample
alignment isn't available.

## What this code does

1. **Loads three modality datasets** (`disk_dataset_reduced.xlsx`, `memory_dataset_reduced.xlsx`,
   `network_dataset_reduced.xlsx`), auto-detecting each file's sample-identifier column
   (e.g. SHA-256 hash) via a scored heuristic (`detect_id_column`), with manual override flags
   (`--disk-id-column`, etc.) if detection is wrong.

2. **Fuses the three modalities into one dataset**, in one of three modes (`--fusion-mode`):
   - `id` (default logic when possible) — **exact fusion**: only samples whose normalized
     identifier is present in *all three* modalities are kept, feature columns are prefixed
     per-modality, and any sample where the three modalities disagree on class label is
     excluded and logged as a conflict.
   - `target` — **pseudo-fusion**: when no shared identifier exists, samples are instead
     paired up by matching class label only (shuffled within each class). This is clearly
     *not* true multimodal fusion and requires the `--allow-pseudo-fusion` flag to run.
   - `auto` (default) — tries exact `id` fusion first; if that's impossible, it automatically
     falls back to `target` pseudo-fusion but writes it to a separate
     `fused_pseudo_sensitivity/` output folder and a `fusion_identifier_limitation.json`
     explaining why, so the two are never confused in reporting.

3. **Trains and evaluates several classifiers** on the fused dataset: Logistic Regression,
   Random Forest, SVM (RBF), KNN, and (if installed) XGBoost and CatBoost.

4. **Cross-validates properly to avoid leakage**:
   - Uses `StratifiedGroupKFold`/`GroupKFold` (falling back to `StratifiedKFold`) over
     10 folds by default.
   - Per training fold only: missing-value imputation, categorical encoding, per-modality
     feature selection (`SelectKBest` + mutual information, top-N features per stream),
     scaling (for distance/gradient-based models), and SMOTE oversampling are all fit fresh
     — never on the full dataset — to prevent test-set information leaking into training.

5. **Builds a weighted ensemble** across the trained models, with the per-model weight
   exponent (`alpha`) either fixed or selected automatically via inner cross-validation
   (`--fused-ensemble-alpha`), and runs diagnostics on it.

6. **Runs additional analyses**:
   - Modality ablation: single/pair/all-modality subsets to see which combination of
     disk/memory/network contributes most (`run_fused_modality_ablation`).
   - Class-imbalance comparison: no resampling vs. training-fold-only SMOTE.
   - Optional chronological (temporal) train/test split if a verified timestamp column
     is supplied — never fabricated if one isn't provided.
   - Explainability study: SHAP, LIME, and Sparse Representation Classification (SRC)-based
     feature attributions on a chosen tree-based model, with perturbation-stability metrics
     (Jaccard@k, Spearman rank correlation, cosine similarity) to test how stable the
     explanations are under small Gaussian perturbations.

7. **Reports results with statistical rigor**: bootstrap confidence intervals, paired
   sign-permutation tests between models, and Holm–Bonferroni multiple-comparison correction.

8. **Writes a full audit trail** to the output directory: alignment/conflict logs, feature
   provenance manifest, class-balance report, per-fold and per-class metrics, confusion
   matrices, ROC curves, ensemble diagnostics, reproducibility report (library versions,
   platform, timing), a "reviewer response matrix", and an execution manifest — so every
   run is traceable end to end.

## Key design principles enforced in the code

- **No fake multimodal fusion presented as real fusion.** Target-label pseudo-pairing is
  always segregated into its own output folder and labelled a sensitivity analysis only.
- **No data leakage.** Every preprocessing step (imputation, encoding, feature selection,
  scaling, SMOTE) is fit strictly within each training fold.
- **No fabricated results.** If a verified timestamp isn't supplied, temporal evaluation is
  skipped and explicitly logged as not run rather than faked.
- **Cross-platform stability.** Matplotlib is forced to the non-GUI `Agg` backend to avoid
  Tk/Tcl crashes on Windows.

## Usage

```bash
python fused_experiment_code.py --data-dir /path/to/reduced_datasets --cv-folds 10
```

Useful flags:
- `--fusion-mode {auto,id,target}` — controls fusion strategy (see above).
- `--allow-pseudo-fusion` — required to explicitly request `target` mode.
- `--use-gpu` — enables GPU for XGBoost/CatBoost if available.
- `--skip-fused-ablation`, `--skip-fused-imbalance-comparison` — skip optional analyses for a faster run.
- `--fused-time-column` — enables chronological evaluation if a verified timestamp exists.
- `--explain-samples`, `--explain-perturbations` — control the size/cost of the explainability study.

Run `python fused_experiment_code.py --help` for the full argument list.

## Output structure

```
results_fused_reviewer_updated/
├── construction/                      # candidate ID-column detection logs per modality
├── plots/                             # class counts, confusion matrices, ROC curves, fold boxplots
├── fused_dataset.csv / .xlsx          # the fused feature table actually evaluated
├── fused_alignment_audit.csv          # per-modality + three-way alignment statistics
├── fused_excluded_label_conflicts.csv # samples dropped for cross-modality label disagreement
├── fusion_method.txt                  # which fusion method was actually used
├── fused_pseudo_sensitivity/          # (only if exact fusion wasn't possible) pseudo-fusion results
├── validation_summary.json
├── execution_manifest.json
└── ... (per-model results, ensemble diagnostics, ablation, imbalance comparison, explainability reports)
```

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.
Optional (feature-gated, code degrades gracefully if missing): `imbalanced-learn` (SMOTE),
`xgboost`, `catboost`, `shap`, `lime`.
