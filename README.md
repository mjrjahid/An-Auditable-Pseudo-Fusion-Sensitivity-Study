
## An Auditable Study of Malware Family Attribution Across Disk, Memory, and Network Artifacts

This repository contains the dataset, experimental code, and supporting results for an audit-first study of malware-family attribution using disk, memory, and network evidence.

The central question is not only whether a classifier achieves high accuracy, but whether the evidence streams are aligned strongly enough to support a genuine multimodal claim. The supplied reduced source tables do not contain an accepted common sample identifier. Therefore, the current benchmark is explicitly treated as a **family-conditioned pseudo-fusion sensitivity study**, not as verified sample-level multimodal fusion.

> [!IMPORTANT]
> Results produced from target-label pairing must not be described as evidence of sample-level disk–memory–network complementarity. Genuine multimodal fusion requires the same verified identifier—preferably SHA-256 or an equivalent execution-lineage identifier—to link all three evidence streams.

## Study Overview

The evaluated cohort contains:

- **1,044** family-conditioned pseudo-fused records;
- **18** Windows malware families;
- **21** predictors: seven each from disk, memory, and network evidence;
- **six** candidate classifiers;
- leakage-controlled **10-fold cross-validation**; and
- explanation-stability analysis using **SHAP, LIME, and class-conditioned sparse representation (SRC)**.

The framework audits identifier quality before constructing the benchmark. If valid common identifiers exist, the code performs exact sample-level fusion. Otherwise, the default `auto` mode isolates the label-conditioned experiment in a clearly named pseudo-sensitivity output directory.

## Methodological Contributions

1. **Identifier audit:** distinguishes exact sample-level fusion from label-conditioned pseudo-fusion.
2. **Leakage-controlled evaluation:** fits imputation, encoding, feature selection, scaling, and SMOTE only on training folds.
3. **Nested ensemble analysis:** selects ensemble members and weights using inner cross-validation within each outer training fold.
4. **Error-diversity diagnostics:** evaluates model-error correlation and prediction disagreement.
5. **Explainability stability:** compares SHAP, LIME, and class-conditioned SRC under controlled perturbations.
6. **Bounded forensic interpretation:** separates research sensitivity evidence from operational sample-level malware attribution.

## Experimental Pipeline

1. Load the reduced disk, memory, and network feature tables.
2. Detect and audit candidate sample-identifier columns.
3. Construct either:
   - an exact-ID-aligned fused cohort; or
   - an explicitly labelled target-paired pseudo-fusion sensitivity cohort.
4. Remove families with fewer than the configured minimum number of observations.
5. Run stratified grouped outer cross-validation.
6. Within each training fold:
   - median-impute numeric variables;
   - constant-impute and ordinal-encode categorical variables;
   - select up to eight features per evidence stream using mutual information;
   - standardize scale-sensitive models; and
   - apply SMOTE only to the transformed training partition.
7. Evaluate individual models and a nested top-three soft-voting ensemble.
8. Generate class-level results, modality diagnostics, error-diversity analyses, explainability outputs, and reproducibility reports.

## Models

The script evaluates:

- Logistic Regression;
- Random Forest;
- RBF-SVM;
- K-Nearest Neighbors;
- XGBoost, when installed; and
- CatBoost, when installed.

Random Forest is used as the preferred tree-based model for the explainability study when available. Ensemble membership, weighting, and the weight exponent are determined without consulting outer test-fold performance.

## Repository Structure

```text
.
├── fused_experiment_code.py
├── Dataset/
│   ├── disk_dataset_reduced.xlsx
│   ├── memory_dataset_reduced.xlsx
│   └── network_dataset_reduced.xlsx
├── fused_dataset.csv
├── results_fused_reviewer_updated/
└── README.md
```

`fused_dataset.csv` is the processed benchmark released for inspection and secondary analysis. The current experiment script reconstructs the benchmark from the three reduced Excel source tables; those tables are required to reproduce the complete identifier audit and dataset-construction procedure.

Before uploading the Python file, rename `fused_experiment_code(1).py` to the cleaner repository name `fused_experiment_code.py`.

## Input Requirements

Each reduced Excel table must:

- use one of the exact filenames shown above;
- contain a `target` column with the malware-family label; and
- optionally contain a shared sample identifier such as `sha256_sample`, `sha256`, `sample_hash`, `hash`, `sample_id`, or `id`.

If automatic identifier detection selects the wrong column, provide the column names explicitly through:

```text
--disk-id-column
--memory-id-column
--network-id-column
```

By default, families with fewer than 20 observations are removed independently from each source table before cohort construction.

## Installation

Python **3.10 or later** is recommended.

```bash
git clone https://github.com/mjrjahid/An-Auditable-Pseudo-Fusion-Sensitivity-Study.git
cd An-Auditable-Pseudo-Fusion-Sensitivity-Study

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib scikit-learn imbalanced-learn openpyxl xgboost catboost shap lime
```

XGBoost, CatBoost, SHAP, and LIME are optional in the source code, but installing them is recommended to reproduce the complete model comparison and explainability study.

## Running the Experiment

### Default audit-first run

```bash
python fused_experiment_code.py \
  --data-dir Dataset \
  --output-dir results_fused_reviewer_updated
```

The default `--fusion-mode auto` behavior is:

- use exact identifier alignment when it can be established; or
- otherwise run a separately stored pseudo-fusion sensitivity analysis and generate an identifier-limitation report.

### Require exact sample-level alignment

```bash
python fused_experiment_code.py \
  --data-dir Dataset \
  --output-dir results_exact_id \
  --fusion-mode id \
  --disk-id-column sha256 \
  --memory-id-column sha256 \
  --network-id-column sha256
```

This mode stops with an error if exact cross-stream alignment cannot be established.

### Explicit pseudo-fusion sensitivity run

```bash
python fused_experiment_code.py \
  --data-dir Dataset \
  --output-dir results_pseudo_sensitivity \
  --fusion-mode target \
  --allow-pseudo-fusion
```

The acknowledgement flag is required because target-conditioned pairing is not genuine sample-level multimodal fusion.

### Faster exploratory run

```bash
python fused_experiment_code.py \
  --data-dir Dataset \
  --output-dir results_quick \
  --skip-fused-ablation \
  --skip-fused-imbalance-comparison \
  --explain-samples 20 \
  --explain-perturbations 5
```

### Optional GPU acceleration

```bash
python fused_experiment_code.py \
  --data-dir Dataset \
  --output-dir results_gpu \
  --use-gpu
```

GPU execution requires compatible XGBoost/CatBoost installations and system drivers.

## Important Command-Line Options

| Option | Purpose | Default |
|---|---|---:|
| `--cv-folds` | Number of outer cross-validation folds | `10` |
| `--min-samples` | Minimum observations retained per family | `20` |
| `--fusion-mode` | Fusion policy: `auto`, `id`, or `target` | `auto` |
| `--fused-ensemble-alpha` | Fixed ensemble exponent or leakage-safe inner-CV selection | `auto` |
| `--explain-samples` | Maximum held-out samples explained | `100` |
| `--explain-perturbations` | Perturbations generated per explained sample | `20` |
| `--explain-perturb-sigma` | Gaussian perturbation scale relative to feature standard deviation | `0.02` |
| `--fused-time-column` | Verified timestamp column for chronological evaluation | not set |
| `--sandbox-metadata-json` | Optional verified execution-environment metadata | not set |
| `--fused-feature-rationale-json` | Optional mapping from features to scientific rationales | not set |

Run the following command to display every option:

```bash
python fused_experiment_code.py --help
```

## Main Outputs

The selected output directory contains CSV, Excel, JSON, text, and PNG artifacts. Important files include:

| Output | Description |
|---|---|
| `fusion_method.txt` | Records whether exact-ID or target-paired construction was used |
| `fusion_identifier_limitation.json` | Documents why exact alignment was unavailable |
| `fused_alignment_audit.csv` | Reports exact-identifier alignment evidence when applicable |
| `fused_dataset.csv` | Constructed fused or pseudo-fused cohort |
| `fused_feature_manifest.csv` | Feature provenance and evidence-stream assignment |
| `fused_training_fold_feature_selection.csv` | Fold-level mutual-information selection records |
| `*_all_model_results.csv` | Cross-validated metrics for the individual classifiers |
| `*_ensemble_metrics.csv` | Nested soft-voting ensemble performance |
| `fused_ensemble_alpha_sensitivity.csv` | Ensemble exponent sensitivity analysis |
| `fused_model_error_correlation.csv` | Pairwise model-error correlations |
| `fused_model_diversity.csv` | Pairwise disagreement and correctness diagnostics |
| `fused_imbalance_strategy_comparison.csv` | No-resampling versus training-fold SMOTE comparison |
| `explainability/*_explainability_summary.csv` | SHAP, LIME, and SRC stability summaries |
| `experiment_execution_manifest.csv` | Indicates which requested analyses completed |
| `fused_reproducibility_and_limitations.json` | Environment, parameters, controls, and limitations |

When exact alignment is unavailable, the primary sensitivity outputs are stored under `fused_pseudo_sensitivity/`.

## Reported Sensitivity Results

The released experiment uses family-conditioned pseudo-fusion because no accepted common identifier was available across the three reduced source tables.

| Method | Accuracy | Balanced accuracy | Weighted F1 | Macro F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8802 | 0.7859 | 0.8741 | 0.7889 | 0.8662 |
| Nested top-three soft-voting ensemble | 0.8706 | 0.7766 | 0.8648 | 0.7704 | 0.8557 |

Random Forest produced the strongest overall classification result. The nested ensemble did not improve on it because the strongest tree-based members exhibited correlated errors and limited prediction diversity. In the implemented perturbation experiment, SHAP produced the most stable explanations among SHAP, LIME, and SRC.

These values measure separability in a family-conditioned pseudo-fused feature space. They do **not** demonstrate the benefit of combining disk, memory, and network evidence from the same malware sample or execution.

## Reproducibility Controls

- Global random seed: `42`.
- Outer validation: stratified group cross-validation when usable groups are available.
- Preprocessing, feature selection, scaling, and SMOTE are fitted inside training folds only.
- Ensemble model selection and weighting use inner cross-validation within each outer training fold.
- Held-out folds are used exclusively for evaluation.
- The script records arguments, package versions, platform details, model hyperparameters, timestamps, and execution status.
- Plotting uses a non-interactive backend for reproducible execution on desktop and server systems.

## Limitations

- The released benchmark lacks verified common SHA-256 or execution-lineage identifiers across all evidence streams.
- Generated fusion identifiers prevent a constructed row from appearing in multiple folds, but they cannot establish sample-lineage separation.
- Pseudo-fusion performance cannot establish sample-level forensic complementarity.
- Aligned modality ablation is meaningful only when exact cross-stream alignment exists.
- Chronological robustness is not evaluated unless a verified timestamp is supplied.
- Execution-environment and sandbox-evasion claims require verified external metadata.
- The results should not be generalized to unseen malware families, benign software, or operational deployment without independent validation.

## Citation

If you use the dataset, code, or experimental framework, please cite the associated study:

> **Trust Before Fusion: An Auditable Study of Malware Family Attribution Across Disk, Memory, and Network Artifacts**

Formal citation and BibTeX metadata can be added here after the paper is published. Until then, please cite this repository and include the repository access date.

## Issues and Contributions

Questions, reproducibility reports, and suggested improvements are welcome through the repository’s GitHub Issues page. Contributions should preserve the distinction between exact identifier-aligned fusion and target-conditioned pseudo-fusion.

