""" fused-dataset experiment:
- loads disk, memory, and network records only to construct the fused benchmark
- evaluates only the exact-ID-aligned fused cohort and its aligned evidence ablations
- builds a genuine fused benchmark only from exact normalized sample identifiers
- in default auto mode, stores target-based pseudo-pairing only as a separately
  labelled sensitivity analysis when exact identifiers are unavailable
- writes fused-only alignment, feature-provenance, imbalance, ablation,
  ensemble-sensitivity, explainability, and reproducibility reports
- fits imputation, categorical encoding, feature selection, scaling, and SMOTE
  only on each training fold
- keeps non-GUI matplotlib backend to avoid tkinter/Tcl crashes on Windows

IMPORTANT:
Target-based fusion is NOT the same as true sample-wise fusion and must not be
reported as a genuinely multimodal sample-level benchmark.
"""

import argparse

import copy

from datetime import datetime, timezone

import importlib.metadata

import json

import os

import platform

import re

import sys

import time

from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

plt.ioff()

import numpy as np

import pandas as pd

from scipy import stats

from sklearn.base import clone

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import SparseCoder

from sklearn.metrics import average_precision_score, matthews_corrcoef

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

try:
    from sklearn.model_selection import GroupKFold
except Exception:
    GroupKFold = None

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, label_binarize

from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

RANDOM_STATE = 42

DEFAULT_WINDOWS_DATA_DIR = Path(r"H:\KFUPM\Thesis\dataset\DATASET NEW\newly preprocess")

DEFAULT_MIN_SAMPLES = 20

FUSION_MODALITIES = ("disk", "memory", "network")

FUSED_ENSEMBLE_ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)

FEATURES_PER_STREAM = 8

DATASET_FILES = {
    "disk": "disk_dataset_reduced.xlsx",
    "memory": "memory_dataset_reduced.xlsx",
    "network": "network_dataset_reduced.xlsx",
}

ID_COLUMNS = ["sha256_sample", "sha256", "sample_hash", "hash", "sample_id", "id"]

METRIC_LABELS = {
    "acc": "Accuracy",
    "prec": "Precision (weighted)",
    "rec": "Recall (weighted)",
    "f1": "F1 (weighted)",
    "auc": "AUC (OVR)",
}

def print_header(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in str(s)).strip().replace(" ", "_")

def bootstrap_ci(values, ci: float = 95.0, n_boot: int = 2000, random_state: int = RANDOM_STATE):
    """Return mean, std, and bootstrap percentile confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) == 1:
        return mean, std, mean, mean
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100.0 - alpha))
    return mean, std, lo, hi

def format_mean_std_ci(mean, std, lo, hi):
    if np.isnan(mean):
        return "nan"
    return f"{mean:.4f} ± {std:.4f} [95% CI {lo:.4f}, {hi:.4f}]"

def paired_sign_permutation_pvalue(left, right, n_permutations=10000, random_state=RANDOM_STATE):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    differences = left[mask] - right[mask]
    if differences.size < 2:
        return np.nan
    observed = abs(float(np.mean(differences)))
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(random_state)
    extreme = 0
    for _ in range(int(n_permutations)):
        signs = rng.choice([-1.0, 1.0], size=differences.size)
        extreme += abs(float(np.mean(differences * signs))) >= observed
    return float((extreme + 1) / (int(n_permutations) + 1))

def holm_adjust_pvalues(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(values))[0]
    if valid_idx.size == 0:
        return adjusted
    order = valid_idx[np.argsort(values[valid_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, float(values[idx]) * (m - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted

def build_cv_splitter(y, groups=None, n_splits=10):
    """Prefer stratified grouped splits when a usable group vector exists."""
    y = np.asarray(y)
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        effective_splits = min(int(n_splits), int(n_groups), min_class_count)
        if effective_splits < 2:
            raise ValueError("Need at least two groups and two samples in every class for grouped validation.")
        if StratifiedGroupKFold is not None:
            return StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE), \
                f"StratifiedGroupKFold(n_splits={effective_splits})"
        if GroupKFold is not None:
            return GroupKFold(n_splits=effective_splits), f"GroupKFold(n_splits={effective_splits})"
    effective_splits = min(int(n_splits), min_class_count)
    if effective_splits < 2:
        raise ValueError("Need at least two samples in every class for stratified validation.")
    return StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE), \
        f"StratifiedKFold(n_splits={effective_splits})"

def iter_splits(splitter, X, y, groups=None):
    if groups is not None:
        try:
            return splitter.split(X, y, groups)
        except TypeError:
            return splitter.split(X, y)
    return splitter.split(X, y)

def safe_clone(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)

def build_search_dirs(cli_data_dir=None):
    script_dir = Path(__file__).resolve().parent
    dirs = []
    if cli_data_dir:
        dirs.append(Path(cli_data_dir))
    dirs.extend([
        DEFAULT_WINDOWS_DATA_DIR,
        script_dir / "Dataset",
        script_dir,
        Path.cwd() / "Dataset",
        Path.cwd(),
    ])
    unique = []
    seen = set()
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique

def find_existing_file(search_dirs, filename):
    checked = []
    for d in search_dirs:
        candidate = d / filename
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}\n\nChecked locations:\n  - " + "\n  - ".join(checked)
    )

def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def looks_like_sha_series(series: pd.Series) -> float:
    try:
        s = series.dropna().astype(str).str.strip()
        if s.empty:
            return 0.0
        patt = r"^[a-fA-F0-9]{64}$"
        return float(s.str.match(patt).mean())
    except Exception:
        return 0.0

def detect_id_column(df: pd.DataFrame, explicit_name=None):
    if explicit_name and explicit_name in df.columns:
        return explicit_name, pd.DataFrame([{
            "column": explicit_name,
            "score": 999.0,
            "reason": "explicit override",
        }])

    rows = []
    for col in df.columns:
        norm = normalize_name(col)
        score = 0.0
        reasons = []

        if norm in {normalize_name(x) for x in ID_COLUMNS}:
            score += 100
            reasons.append("exact normalized id-name match")

        if "sha256" in norm:
            score += 90
            reasons.append("contains sha256")

        if "sha" in norm:
            score += 30
            reasons.append("contains sha")

        if "sample" in norm:
            score += 20
            reasons.append("contains sample")

        if norm.endswith("id") or "id" in norm:
            score += 10
            reasons.append("contains id")

        sha_ratio = looks_like_sha_series(df[col])
        score += 100 * sha_ratio
        if sha_ratio > 0:
            reasons.append(f"{sha_ratio:.2%} values look like SHA-256")

        nunique_ratio = 0.0
        try:
            s = df[col].dropna()
            if len(s):
                nunique_ratio = s.astype(str).nunique() / len(s)
        except Exception:
            pass
        score += 5 * nunique_ratio
        if nunique_ratio > 0.5:
            reasons.append("high uniqueness")

        rows.append({
            "column": col,
            "score": score,
            "reason": "; ".join(reasons) if reasons else "no strong signal",
        })

    candidate_df = pd.DataFrame(rows).sort_values(["score", "column"], ascending=[False, True]).reset_index(drop=True)
    best_col = candidate_df.iloc[0]["column"] if len(candidate_df) and candidate_df.iloc[0]["score"] >= 40 else None
    return best_col, candidate_df

def model_requires_scaling(model_name: str) -> bool:
    scale_names = ("Logistic Regression", "SVM", "KNN", "Naive Bayes")
    return any(key in model_name for key in scale_names)

def safe_smote_fit_resample(X_train, y_train, requested_k=5):
    if SMOTE is None:
        raise RuntimeError(
            "Training-fold SMOTE requires imbalanced-learn. Install it with: "
            "pip install imbalanced-learn"
        )
    class_counts = pd.Series(y_train).value_counts()
    min_count = int(class_counts.min()) if len(class_counts) else 0
    if min_count <= 1:
        print("[WARN] SMOTE skipped for this fold because a class has <= 1 sample.")
        return X_train, y_train
    k_neighbors = min(requested_k, min_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=max(1, k_neighbors))
    return smote.fit_resample(X_train, y_train)

def fit_fold_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit missing-value handling and categorical encoding on training data only."""
    X_train = pd.DataFrame(X_train).copy().replace([np.inf, -np.inf], np.nan)
    X_test = pd.DataFrame(X_test).copy().replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("No usable fused features remain after excluding identifiers and target labels.")
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    X_test_processed = np.asarray(preprocessor.transform(X_test), dtype=float)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    return X_train_processed, X_test_processed, feature_names, preprocessor

def select_features_by_stream_training_only(X_train, X_test, y_train, feature_names, features_per_stream=8):
    """Select at most k features per evidence stream using training-fold mutual information."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    feature_names = np.asarray(feature_names, dtype=str)
    if features_per_stream is None or int(features_per_stream) <= 0:
        selected = np.arange(len(feature_names), dtype=int)
        return X_train, X_test, feature_names, pd.DataFrame({
            "Feature": feature_names, "Evidence stream": [n.split("__", 1)[0] for n in feature_names],
            "Selected": True, "Selection reason": "Feature selection disabled",
        })

    groups = {}
    for idx, name in enumerate(feature_names):
        stream = name.split("__", 1)[0] if "__" in name else "unassigned"
        groups.setdefault(stream, []).append(idx)

    selected_indices = []
    rows = []
    for stream, indices in groups.items():
        indices = np.asarray(indices, dtype=int)
        k = min(int(features_per_stream), len(indices))
        if len(indices) > k and len(np.unique(y_train)) > 1:
            selector = SelectKBest(
                score_func=lambda values, labels: mutual_info_classif(
                    values, labels, random_state=RANDOM_STATE
                ),
                k=k,
            )
            selector.fit(X_train[:, indices], y_train)
            support = selector.get_support()
            scores = np.asarray(selector.scores_, dtype=float)
        else:
            support = np.ones(len(indices), dtype=bool)
            scores = np.full(len(indices), np.nan)
        selected_indices.extend(indices[support].tolist())
        for local_idx, global_idx in enumerate(indices):
            rows.append({
                "Feature": str(feature_names[global_idx]),
                "Evidence stream": stream,
                "Mutual information (training fold only)": float(scores[local_idx]) if np.isfinite(scores[local_idx]) else np.nan,
                "Selected": bool(support[local_idx]),
                "Selection reason": (
                    f"Top-{k} within {stream} on training fold" if len(indices) > k
                    else f"All {len(indices)} supplied {stream} features retained"
                ),
            })
    selected_indices = np.asarray(sorted(selected_indices), dtype=int)
    return (
        X_train[:, selected_indices],
        X_test[:, selected_indices],
        feature_names[selected_indices],
        pd.DataFrame(rows),
    )

def prepare_fold_data(X_train, X_test, y_train, model_name: str, smote_k: int,
                      apply_smote: bool = True, features_per_stream: int | None = FEATURES_PER_STREAM,
                      return_artifacts: bool = False):
    X_train_processed, X_test_processed, feature_names, preprocessor = fit_fold_preprocessor(
        X_train, X_test
    )
    X_train_selected, X_test_selected, selected_names, selection_df = (
        select_features_by_stream_training_only(
            X_train_processed, X_test_processed, y_train, feature_names,
            features_per_stream=features_per_stream,
        )
    )
    scaler = None
    if model_requires_scaling(model_name):
        scaler = StandardScaler()
        X_train_base = scaler.fit_transform(X_train_selected)
        X_test_eval = scaler.transform(X_test_selected)
    else:
        X_train_base = X_train_selected
        X_test_eval = X_test_selected

    if apply_smote:
        X_res, y_res = safe_smote_fit_resample(X_train_base, y_train, requested_k=smote_k)
    else:
        X_res, y_res = X_train_base, np.asarray(y_train)
    if not return_artifacts:
        return X_res, y_res, X_test_eval
    artifacts = {
        "preprocessor": preprocessor,
        "scaler": scaler,
        "selected_feature_names": list(selected_names.astype(str)),
        "selection_df": selection_df,
        "X_train_selected_unscaled": np.asarray(X_train_selected, dtype=float),
        "X_test_selected_unscaled": np.asarray(X_test_selected, dtype=float),
        "X_train_model_space": np.asarray(X_train_base, dtype=float),
        "X_test_model_space": np.asarray(X_test_eval, dtype=float),
    }
    return X_res, y_res, X_test_eval, artifacts

def compute_metrics(y_true, y_pred, y_prob, labels=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
        elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auc_val = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_val = np.nan
    except Exception:
        auc_val = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, prec, rec, f1_w, f1_macro, auc_val, cm

def compute_metrics_ensemble(y_true, y_pred, y_prob, labels=None):
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
        elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auc_ovr = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_ovr = np.nan
    except Exception:
        auc_ovr = np.nan
    return acc, prec_w, rec_w, f1_w, f1_macro, auc_ovr

def compute_metrics_extended(y_true, y_pred, y_prob, labels=None):
    """Return the classifier metrics requested for the explainability base model."""
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
            y_bin = label_binarize(y_true, classes=labels if labels is not None else np.unique(y_true))
            auprc = average_precision_score(y_bin, y_prob, average="macro")
        elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auroc = roc_auc_score(y_true, y_prob[:, 1])
            auprc = average_precision_score(y_true, y_prob[:, 1])
        else:
            auroc = np.nan
            auprc = np.nan
    except Exception:
        auroc = np.nan
        auprc = np.nan

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
        "auprc": float(auprc) if np.isfinite(auprc) else np.nan,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "mcc": float(mcc),
    }

def _extract_class_specific_values(values, class_idx: int):
    if values is None:
        return None
    if isinstance(values, list):
        if class_idx < len(values):
            return np.asarray(values[class_idx])
        return np.asarray(values[0])
    arr = np.asarray(values)
    if arr.ndim == 3:
        if class_idx < arr.shape[-1]:
            return arr[..., class_idx]
        return arr[..., 0]
    return arr

def _rank_to_scores(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float).ravel()
    out = np.abs(vec)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _topk_set(vector: np.ndarray, k: int = 10) -> set:
    scores = _rank_to_scores(vector)
    if scores.size == 0:
        return set()
    k = min(k, scores.size)
    return set(np.argsort(scores)[-k:])

def jaccard_at_k(original: np.ndarray, perturbed: np.ndarray, k: int = 10) -> float:
    a = _topk_set(original, k)
    b = _topk_set(perturbed, k)
    if not a and not b:
        return np.nan
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))

def spearman_rank_similarity(original: np.ndarray, perturbed: np.ndarray) -> float:
    orig = _rank_to_scores(original)
    pert = _rank_to_scores(perturbed)
    if orig.size == 0 or pert.size == 0:
        return np.nan
    corr = stats.spearmanr(orig, pert).correlation
    return float(corr) if corr is not None and np.isfinite(corr) else np.nan

def cosine_similarity_score(original: np.ndarray, perturbed: np.ndarray) -> float:
    orig = _rank_to_scores(original).reshape(1, -1)
    pert = _rank_to_scores(perturbed).reshape(1, -1)
    if orig.size == 0 or pert.size == 0:
        return np.nan
    try:
        return float(cosine_similarity(orig, pert)[0, 0])
    except Exception:
        return np.nan

def choose_tree_explainable_model(models: dict) -> str:
    """Choose a pre-specified explainable model without consulting held-out results."""
    candidates = [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "CatBoost",
    ]
    for candidate in candidates:
        if candidate in models:
            return candidate
    return str(next(iter(models)))

def get_model_from_builder(model_name: str, models: dict):
    if model_name in models:
        return models[model_name]
    for key, model in models.items():
        if key.lower() == model_name.lower():
            return model
    raise KeyError(f"Could not find model '{model_name}' in the model builder output.")

def sample_balanced_indices(y: np.ndarray, max_samples: int, random_state: int = RANDOM_STATE):
    y = np.asarray(y)
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y))
    rng = np.random.default_rng(random_state)
    unique = np.unique(y)
    per_class = max(1, max_samples // len(unique))
    chosen = []
    leftovers = []
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        take = min(per_class, len(cls_idx))
        chosen.extend(cls_idx[:take].tolist())
        leftovers.extend(cls_idx[take:].tolist())
    if len(chosen) < max_samples and leftovers:
        leftovers = np.asarray(leftovers, dtype=int)
        rng.shuffle(leftovers)
        remaining = max_samples - len(chosen)
        chosen.extend(leftovers[:remaining].tolist())
    chosen = np.array(sorted(set(chosen)), dtype=int)
    return chosen[:max_samples]

def build_src_dictionaries(X_train_scaled: np.ndarray, y_train: np.ndarray, class_labels: np.ndarray,
                           max_atoms_per_class: int = 50, random_state: int = RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    class_dicts = {}
    global_dict = np.asarray(X_train_scaled, dtype=float)
    for cls in class_labels:
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) == 0:
            continue
        rng.shuffle(cls_idx)
        n_atoms = min(max_atoms_per_class, len(cls_idx))
        class_dicts[int(cls)] = np.asarray(X_train_scaled[cls_idx[:n_atoms]], dtype=float)
    return class_dicts, global_dict

def src_explain_sample(x_scaled: np.ndarray, predicted_class: int, class_dicts: dict, global_dict: np.ndarray,
                       n_nonzero_coefs: int = 10):
    x_scaled = np.asarray(x_scaled, dtype=float).reshape(1, -1)
    dictionary = class_dicts.get(int(predicted_class))
    fallback_used = False
    if dictionary is None or len(dictionary) == 0:
        dictionary = global_dict
        fallback_used = True
    dictionary = np.asarray(dictionary, dtype=float)
    n_nonzero_coefs = max(1, min(int(n_nonzero_coefs), dictionary.shape[0]))
    coder = SparseCoder(dictionary=dictionary, transform_algorithm="omp", transform_n_nonzero_coefs=n_nonzero_coefs)
    code = coder.transform(x_scaled)
    recon = code @ dictionary
    explanation = np.abs(x_scaled - recon).ravel()
    return explanation, recon.ravel(), fallback_used

def shap_explain_sample(model, x_row: np.ndarray, class_idx: int, explainer=None):
    if shap is None:
        raise RuntimeError("shap is not installed.")
    explainer = explainer or shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row.reshape(1, -1))
    vals = _extract_class_specific_values(shap_values, class_idx)
    if vals is None:
        raise RuntimeError("Could not extract SHAP values.")
    vals = np.asarray(vals)
    if vals.ndim == 2:
        vals = vals[0]
    return _rank_to_scores(vals)

def lime_explain_sample(model, x_row: np.ndarray, X_train: np.ndarray, feature_names, class_names, class_idx: int, explainer=None):
    if LimeTabularExplainer is None:
        raise RuntimeError("lime is not installed.")
    explainer = explainer or LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        sample_around_instance=True,
        random_state=RANDOM_STATE,
    )
    exp = explainer.explain_instance(
        data_row=np.asarray(x_row, dtype=float),
        predict_fn=model.predict_proba,
        labels=[int(class_idx)],
        num_features=len(feature_names),
        num_samples=1000,
    )
    scores = np.zeros(len(feature_names), dtype=float)
    local_map = exp.as_map().get(int(class_idx), [])
    for feat_idx, weight in local_map:
        if 0 <= int(feat_idx) < len(scores):
            scores[int(feat_idx)] = float(weight)
    return _rank_to_scores(scores)

def build_perturbation_noise_scale(X_train: np.ndarray):
    std = np.std(np.asarray(X_train, dtype=float), axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std == 0), 1.0, std)
    return std

def _safe_nanmean(series_like):
    arr = np.asarray(series_like, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))

def run_explainability_study(*, dataset_name: str, X: pd.DataFrame, y: np.ndarray, class_names, models: dict,
                             results_df: pd.DataFrame, output_dir: Path, test_size: float = 0.2,
                             max_samples: int = 100, perturbations: int = 20, perturb_sigma: float = 0.02,
                             src_atoms_per_class: int = 50, src_nonzero_coefs: int = 10):
    out_dir = ensure_dir(output_dir)
    explain_dir = ensure_dir(out_dir / "explainability")
    chosen_model_name = choose_tree_explainable_model(models)
    base_model = get_model_from_builder(chosen_model_name, models)
    print_header(f"EXPLAINABILITY BASE MODEL - {dataset_name}")
    print(f"Pre-specified classifier (not selected using holdout results): {chosen_model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.reset_index(drop=True),
        np.asarray(y),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_res, y_res, X_test_eval, fold_artifacts = prepare_fold_data(
        X_train,
        X_test,
        y_train,
        model_name=chosen_model_name,
        smote_k=1,
        return_artifacts=True,
    )
    feature_names = list(fold_artifacts["selected_feature_names"])
    X_train_model = fold_artifacts["X_train_model_space"]
    X_test_model = fold_artifacts["X_test_model_space"]
    selection_df = fold_artifacts["selection_df"].copy()
    save_df(selection_df, explain_dir / "fused_explainability_holdout_feature_selection.csv",
            explain_dir / "fused_explainability_holdout_feature_selection.xlsx")

    model = safe_clone(base_model)
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test_eval)
    y_prob = model.predict_proba(X_test_eval)
    y_prob = align_proba_to_all_classes(y_prob, model.classes_, np.unique(y))
    metrics = compute_metrics_extended(y_test, y_pred, y_prob, labels=np.unique(y))

    metrics_df = pd.DataFrame([{
        "Dataset": dataset_name,
        "Model": chosen_model_name,
        **metrics,
        "test_size": test_size,
        "n_test_samples": int(len(y_test)),
    }])
    save_df(metrics_df, explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_classifier_metrics.csv",
            explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_classifier_metrics.xlsx")

    # Prepare SRC dictionaries using the same training-only transformed feature space.
    scaler = StandardScaler().fit(X_train_model)
    X_train_scaled = scaler.transform(X_train_model)
    X_test_scaled = scaler.transform(X_test_model)
    class_dicts, global_dict = build_src_dictionaries(
        X_train_scaled, y_train, np.unique(y), max_atoms_per_class=src_atoms_per_class, random_state=RANDOM_STATE
    )
    train_noise_scale = build_perturbation_noise_scale(X_train_model)
    shap_explainer = shap.TreeExplainer(model) if shap is not None else None
    lime_explainer = None
    if LimeTabularExplainer is not None:
        lime_explainer = LimeTabularExplainer(
            training_data=np.asarray(X_train_model, dtype=float),
            feature_names=list(feature_names),
            class_names=list(class_names),
            mode="classification",
            discretize_continuous=True,
            sample_around_instance=True,
            random_state=RANDOM_STATE,
        )

    sample_indices = sample_balanced_indices(y_test, max_samples=max_samples, random_state=RANDOM_STATE)
    X_test_sel = X_test_model[sample_indices]
    y_test_sel = y_test[sample_indices]
    y_pred_sel = model.predict(X_test_eval[sample_indices])
    y_prob_sel = model.predict_proba(X_test_eval[sample_indices])
    y_prob_sel = align_proba_to_all_classes(y_prob_sel, model.classes_, np.unique(y))

    per_sample_rows = []
    attribution_rows = []
    fallback_count = 0
    fallback_samples = []

    for idx_local, idx_global in enumerate(sample_indices):
        x_raw = np.asarray(X_test_sel[idx_local], dtype=float)
        pred_class = int(y_pred_sel[idx_local])
        x_src = X_test_scaled[idx_global]
        original = {}
        original["SHAP"] = shap_explain_sample(model, x_raw, pred_class, explainer=shap_explainer) if shap is not None else None
        original["LIME"] = lime_explain_sample(model, x_raw, X_train_model, feature_names, class_names, pred_class, explainer=lime_explainer) if LimeTabularExplainer is not None else None
        src_vec, src_recon, fallback_used = src_explain_sample(
            x_src, pred_class, class_dicts, global_dict, n_nonzero_coefs=src_nonzero_coefs
        )
        original["SRC"] = src_vec
        fallback_count += int(fallback_used)
        if fallback_used:
            fallback_samples.append(int(idx_global))

        for method, vector in original.items():
            if vector is None:
                continue
            scores = _rank_to_scores(vector)
            for feature_idx, score in enumerate(scores):
                feature = feature_names[feature_idx]
                modality = feature.split("__", 1)[0] if "__" in feature else "unassigned"
                attribution_rows.append({
                    "Dataset": dataset_name,
                    "Sample index": int(idx_global),
                    "True class": str(class_names[int(y_test_sel[idx_local])]),
                    "Predicted class": str(class_names[int(pred_class)]),
                    "Correct": bool(int(y_test_sel[idx_local]) == int(pred_class)),
                    "Method": method,
                    "Modality": modality,
                    "Feature": feature,
                    "Importance": float(score),
                })

        perturb_rows = {"SHAP": [], "LIME": [], "SRC": []}
        for rep in range(int(perturbations)):
            noise = np.random.default_rng(RANDOM_STATE + rep + idx_global).normal(
                loc=0.0,
                scale=train_noise_scale * float(perturb_sigma),
                size=x_raw.shape,
            )
            x_pert_raw = x_raw + noise
            x_pert_scaled = scaler.transform(x_pert_raw.reshape(1, -1))[0]
            # class remains the original predicted class for comparison of explanation stability
            if shap is not None:
                try:
                    perturb_rows["SHAP"].append(shap_explain_sample(model, x_pert_raw, pred_class, explainer=shap_explainer))
                except Exception:
                    perturb_rows["SHAP"].append(np.full_like(original["SHAP"], np.nan) if original["SHAP"] is not None else np.nan)
            if LimeTabularExplainer is not None:
                try:
                    perturb_rows["LIME"].append(lime_explain_sample(model, x_pert_raw, X_train_model, feature_names, class_names, pred_class, explainer=lime_explainer))
                except Exception:
                    perturb_rows["LIME"].append(np.full(len(feature_names), np.nan))
            try:
                pert_src, _, pert_fallback = src_explain_sample(x_pert_scaled, pred_class, class_dicts, global_dict,
                                                               n_nonzero_coefs=src_nonzero_coefs)
                perturb_rows["SRC"].append(pert_src)
                fallback_count += int(pert_fallback)
                if pert_fallback:
                    fallback_samples.append(int(idx_global))
            except Exception:
                perturb_rows["SRC"].append(np.full(len(feature_names), np.nan))

        for method in ["SHAP", "LIME", "SRC"]:
            if original.get(method) is None:
                continue
            vals = []
            for pert_vec in perturb_rows[method]:
                if pert_vec is None or not np.isfinite(np.asarray(pert_vec, dtype=float)).any():
                    continue
                vals.append({
                    "Jaccard@10": jaccard_at_k(original[method], pert_vec, k=10),
                    "Spearman": spearman_rank_similarity(original[method], pert_vec),
                    "Cosine": cosine_similarity_score(original[method], pert_vec),
                })
            if vals:
                vals_df = pd.DataFrame(vals)
                scores = _rank_to_scores(original[method])
                top_idx = np.argsort(scores)[::-1][:min(10, len(scores))]
                top_features = "; ".join(
                    f"{feature_names[int(i)]}={scores[int(i)]:.6g}" for i in top_idx
                )
                per_sample_rows.append({
                    "Dataset": dataset_name,
                    "Sample index": int(idx_global),
                    "True class": int(y_test_sel[idx_local]),
                    "Predicted class": int(pred_class),
                    "True class name": str(class_names[int(y_test_sel[idx_local])]),
                    "Predicted class name": str(class_names[int(pred_class)]),
                    "Correct": bool(int(y_test_sel[idx_local]) == int(pred_class)),
                    "Method": method,
                    "Top features": top_features,
                    "Jaccard@10": float(vals_df["Jaccard@10"].mean()),
                    "Spearman": float(vals_df["Spearman"].mean()),
                    "Cosine": float(vals_df["Cosine"].mean()),
                })

    per_sample_df = pd.DataFrame(per_sample_rows)
    save_df(per_sample_df, explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_per_sample.csv",
            explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_per_sample.xlsx")
    if not per_sample_df.empty:
        representative_cases_df = (
            per_sample_df.sort_values(["Correct", "True class name", "Method", "Sample index"])
            .groupby(["Correct", "True class name", "Method"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        save_df(representative_cases_df, explain_dir / "fused_representative_correct_incorrect_explanations.csv",
                explain_dir / "fused_representative_correct_incorrect_explanations.xlsx")

    attribution_df = pd.DataFrame(attribution_rows)
    if not attribution_df.empty:
        feature_summary_df = (
            attribution_df.groupby(["Method", "Feature", "Modality"], as_index=False)["Importance"]
            .mean()
            .rename(columns={"Importance": "Mean absolute importance"})
            .sort_values(["Method", "Mean absolute importance"], ascending=[True, False])
        )
        class_feature_summary_df = (
            attribution_df.groupby(["Method", "True class", "Feature", "Modality"], as_index=False)["Importance"]
            .mean()
            .rename(columns={"Importance": "Mean absolute importance"})
            .sort_values(["Method", "True class", "Mean absolute importance"], ascending=[True, True, False])
        )
        modality_summary_df = (
            attribution_df.groupby(["Method", "Modality"], as_index=False)["Importance"]
            .sum()
            .rename(columns={"Importance": "Total absolute importance"})
        )
        modality_summary_df["Normalized modality importance"] = modality_summary_df.groupby("Method")[
            "Total absolute importance"
        ].transform(lambda s: s / s.sum() if float(s.sum()) else 0.0)
        save_df(feature_summary_df, explain_dir / "fused_global_feature_importance.csv",
                explain_dir / "fused_global_feature_importance.xlsx")
        save_df(class_feature_summary_df, explain_dir / "fused_class_specific_feature_importance.csv",
                explain_dir / "fused_class_specific_feature_importance.xlsx")
        save_df(modality_summary_df, explain_dir / "fused_modality_importance.csv",
                explain_dir / "fused_modality_importance.xlsx")

    summary_rows = []
    for method in ["SHAP", "LIME", "SRC"]:
        method_df = per_sample_df[per_sample_df["Method"] == method].copy()
        if method_df.empty:
            continue
        summary_rows.append({
            "Dataset": dataset_name,
            "Method": method,
            "Mean Jaccard@10": _safe_nanmean(method_df["Jaccard@10"]),
            "Mean Spearman": _safe_nanmean(method_df["Spearman"]),
            "Mean Cosine": _safe_nanmean(method_df["Cosine"]),
            "Samples": int(len(method_df)),
        })
    summary_df = pd.DataFrame(summary_rows)
    save_df(summary_df, explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_summary.csv",
            explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_summary.xlsx")

    fallback_report = {
        "Dataset": dataset_name,
        "Chosen model": chosen_model_name,
        "SRC fallback count": int(fallback_count),
        "SRC fallback rate": float(fallback_count / max(1, len(sample_indices) * max(1, perturbations) + len(sample_indices))),
        "Fallback sample indices": fallback_samples,
        "Interpretation": "Zero fallback means SRC remained class-conditioned and never needed a global dictionary.",
    }
    Path(explain_dir / f"{sanitize_filename(dataset_name.lower())}_src_fallback_report.json").write_text(
        json.dumps(fallback_report, indent=2), encoding="utf-8"
    )

    # Visual summary
    if not summary_df.empty:
        plt.figure(figsize=(9, 4.8))
        x = np.arange(len(summary_df))
        width = 0.25
        plt.bar(x - width, summary_df["Mean Jaccard@10"].astype(float).values, width, label="Jaccard@10")
        plt.bar(x, summary_df["Mean Spearman"].astype(float).values, width, label="Spearman")
        plt.bar(x + width, summary_df["Mean Cosine"].astype(float).values, width, label="Cosine")
        plt.xticks(x, summary_df["Method"].astype(str).values)
        plt.ylim(bottom=0)
        plt.title(f"{dataset_name}: explanation stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(explain_dir / f"{sanitize_filename(dataset_name.lower())}_explainability_stability.png", dpi=300)
        plt.close()

    return {
        "dataset": dataset_name,
        "chosen_model": chosen_model_name,
        "metrics_df": metrics_df,
        "per_sample_df": per_sample_df,
        "attribution_df": attribution_df,
        "summary_df": summary_df,
        "fallback_report": fallback_report,
    }

def build_explainability_conclusion(all_explainability: dict, output_root: Path):
    frames = [v["summary_df"] for v in all_explainability.values() if v.get("summary_df") is not None and not v["summary_df"].empty]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    overall = combined.groupby("Method", as_index=False).agg({
        "Mean Jaccard@10": "mean",
        "Mean Spearman": "mean",
        "Mean Cosine": "mean",
    })
    overall = overall.sort_values(["Mean Jaccard@10", "Mean Spearman", "Mean Cosine"], ascending=False).reset_index(drop=True)
    rows = []
    rows.append("Explainability summary for the fused dataset")
    rows.append("")
    for _, row in overall.iterrows():
        rows.append(
            f"{row['Method']}: Jaccard@10={row['Mean Jaccard@10']:.4f}, Spearman={row['Mean Spearman']:.4f}, Cosine={row['Mean Cosine']:.4f}"
        )
    rows.append("")
    if not overall.empty:
        best_j = overall.sort_values(["Mean Jaccard@10", "Mean Spearman"], ascending=False).iloc[0]["Method"]
        best_s = overall.sort_values(["Mean Spearman", "Mean Jaccard@10"], ascending=False).iloc[0]["Method"]
        best_c = overall.sort_values(["Mean Cosine"], ascending=False).iloc[0]["Method"]
        rows.append(f"Top Jaccard@10 method: {best_j}")
        rows.append(f"Top Spearman method: {best_s}")
        rows.append(f"Top cosine method: {best_c}")
        rows.append("")
        rows.append(
            "SRC is evaluated here as a class-conditioned explainer for exact-ID-aligned multisource malware attribution, "
            "especially when investigators need evidence-pattern resemblance in addition to feature attribution."
        )
        rows.append(
            "In the current run, the class-conditioned SRC pipeline remained on the predicted-class dictionary and the recorded global-dictionary fallback count was zero whenever the class dictionary was available."
        )
    conclusion = "\n".join(rows)
    Path(output_root / "explainability_conclusion.txt").write_text(conclusion, encoding="utf-8")
    save_df(overall, output_root / "explainability_method_comparison.csv", output_root / "explainability_method_comparison.xlsx")
    return conclusion

def align_proba_to_all_classes(proba, est_classes, all_classes):
    aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    for j, c in enumerate(est_classes):
        if c in class_to_idx:
            aligned[:, class_to_idx[c]] = proba[:, j]
    return aligned

def init_per_class_tracker(all_classes):
    return {m: {int(c): [] for c in all_classes} for m in ["precision", "recall", "f1", "support"]}

def update_per_class_tracker(tracker: dict, y_true, y_pred, labels):
    p, r, f1_vals, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for idx, label in enumerate(labels):
        tracker["precision"][int(label)].append(float(p[idx]))
        tracker["recall"][int(label)].append(float(r[idx]))
        tracker["f1"][int(label)].append(float(f1_vals[idx]))
        tracker["support"][int(label)].append(int(s[idx]))

def build_class_accuracy_df(metric_map: dict, class_names, model_name: str, dataset_name: str):
    rows = []
    for class_idx, class_name in enumerate(class_names):
        recall_vals = metric_map.get("recall", {}).get(class_idx, [])
        support_vals = metric_map.get("support", {}).get(class_idx, [])
        mean_acc, std_acc, lo_acc, hi_acc = bootstrap_ci(recall_vals)
        support_arr = np.asarray(support_vals, dtype=float)
        support_arr = support_arr[~np.isnan(support_arr)]
        rows.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Class index": class_idx,
            "Class": str(class_name),
            "Class Accuracy Mean": mean_acc,
            "Class Accuracy Std": std_acc,
            "Class Accuracy CI95 Low": lo_acc,
            "Class Accuracy CI95 High": hi_acc,
            "Class Accuracy (mean ± std)": format_mean_std_ci(mean_acc, std_acc, lo_acc, hi_acc),
            "Mean Support": float(np.nanmean(support_arr)) if len(support_arr) else np.nan,
            "Fold Count": int(len(recall_vals)),
        })
    return pd.DataFrame(rows)

def build_per_class_metrics_df(metric_map: dict, class_names, model_name: str, dataset_name: str):
    rows = []
    for class_idx, class_name in enumerate(class_names):
        row = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Class index": class_idx,
            "Class": str(class_name),
        }
        for metric, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
            mean, std, low, high = bootstrap_ci(metric_map.get(metric, {}).get(class_idx, []))
            row.update({
                f"{label} Mean": mean,
                f"{label} Std": std,
                f"{label} CI95 Low": low,
                f"{label} CI95 High": high,
            })
        supports = np.asarray(metric_map.get("support", {}).get(class_idx, []), dtype=float)
        row["Mean Support"] = float(np.nanmean(supports)) if supports.size else np.nan
        row["Fold Count"] = int(len(metric_map.get("recall", {}).get(class_idx, [])))
        rows.append(row)
    return pd.DataFrame(rows)

def save_df(df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None = None):
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        df.to_excel(xlsx_path, index=False)
    return df

def add_count_labels(bars, fontsize=9):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def plot_class_counts(y_names, out_path: Path, title: str):
    counts = pd.Series(y_names).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    colors = plt.cm.Greys(np.linspace(0.25, 0.85, len(counts)))
    bars = plt.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="black", linewidth=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    add_count_labels(bars)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_binary_or_multiclass_roc(y_true_all, y_prob_all, model_name: str, class_names, out_path: Path):
    plt.figure(figsize=(7, 6))
    unique_classes = np.unique(y_true_all)

    if len(unique_classes) <= 2 and y_prob_all.shape[1] >= 2:
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        Y = label_binarize(y_true_all, classes=np.arange(len(class_names)))
        fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_prob_all.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, linewidth=2.2, label=f"micro-average (AUC = {auc_micro:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def safe_normalize_confusion_matrix(cm):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

def plot_confusion_matrix(cm, class_names, out_path: Path, title: str, normalize: bool = False):
    disp = safe_normalize_confusion_matrix(cm) if normalize else np.asarray(cm, dtype=float)
    n_classes = len(class_names)
    fig_size = (max(8, min(18, 0.45 * n_classes + 4)), max(6, min(18, 0.45 * n_classes + 4)))
    plt.figure(figsize=fig_size)
    im = plt.imshow(disp, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(n_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if n_classes <= 20:
        thresh = float(np.nanmax(disp)) / 2.0 if disp.size else 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                value = disp[i, j]
                if normalize:
                    text = f"{value:.2f}"
                else:
                    text = f"{int(round(value))}"
                plt.text(j, i, text, ha="center", va="center", fontsize=7,
                         color="black" if value < thresh else "white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def build_top_confusions(cm, class_names, top_n: int = 10):
    cm = np.asarray(cm, dtype=float)
    rows = []
    for i, actual in enumerate(class_names):
        row_sum = cm[i].sum()
        for j, predicted in enumerate(class_names):
            if i == j:
                continue
            count = float(cm[i, j])
            if count <= 0:
                continue
            rate = count / row_sum if row_sum else np.nan
            rows.append({
                "Actual": str(actual),
                "Predicted": str(predicted),
                "Count": count,
                "Rate": rate,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["Count", "Rate"], ascending=[False, False]).head(top_n).reset_index(drop=True)

def plot_metric_summary_bars(results_df: pd.DataFrame, out_path: Path, metric_col: str, err_col: str, title: str):
    df = results_df[["Model", metric_col, err_col]].copy().dropna(subset=[metric_col])
    df = df.sort_values(metric_col, ascending=True).reset_index(drop=True)
    plt.figure(figsize=(max(8, 0.55 * len(df) + 4), max(4, 0.4 * len(df) + 2)))
    x = np.arange(len(df))
    vals = df[metric_col].astype(float).values
    errs = df[err_col].astype(float).fillna(0.0).values if err_col in df.columns else np.zeros_like(vals)
    bars = plt.barh(x, vals, xerr=errs, capsize=4)
    plt.yticks(x, df["Model"].astype(str))
    plt.xlabel(metric_col)
    plt.title(title)
    for bar, value in zip(bars, vals):
        plt.text(value, bar.get_y() + bar.get_height()/2, f" {value:.3f}", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_fold_metric_boxplots(fold_df: pd.DataFrame, out_dir: Path, dataset_name: str):
    metric_map = {
        "Accuracy": "Accuracy",
        "F1 (weighted)": "F1 (weighted)",
        "F1 (macro)": "F1 (macro)",
        "AUC (OVR)": "AUC (OVR)",
    }
    for metric_label, metric_col in metric_map.items():
        tmp = fold_df[["Model", metric_col]].dropna()
        if tmp.empty:
            continue
        models = list(tmp["Model"].drop_duplicates())
        data = [tmp.loc[tmp["Model"] == m, metric_col].astype(float).values for m in models]
        plt.figure(figsize=(max(8, 0.65 * len(models) + 4), 5.5))
        plt.boxplot(data, showmeans=True)
        plt.xticks(np.arange(1, len(models) + 1), models, rotation=35, ha="right")
        plt.ylabel(metric_label)
        plt.title(f"{dataset_name}: Fold-level {metric_label}")
        plt.tight_layout()
        plt.savefig(out_dir / f"fold_boxplot_{sanitize_filename(metric_label)}.png", dpi=300)
        plt.close()

def plot_fusion_gain_bar(gain_df: pd.DataFrame, out_path: Path, title: str):
    if gain_df.empty:
        return
    df = gain_df[["Class", "Fusion gain"]].copy().sort_values("Fusion gain", ascending=True)
    plt.figure(figsize=(max(8, 0.55 * len(df) + 4), max(4, 0.35 * len(df) + 2)))
    x = np.arange(len(df))
    bars = plt.barh(x, df["Fusion gain"].astype(float).values)
    plt.yticks(x, df["Class"].astype(str))
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fused accuracy minus best single-source accuracy")
    plt.title(title)
    for bar, value in zip(bars, df["Fusion gain"].astype(float).values):
        plt.text(value, bar.get_y() + bar.get_height()/2, f" {value:+.3f}", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def get_models_for_dataset(dataset_key, use_gpu=False):
    if dataset_key != "fused":
        raise ValueError("This standalone script evaluates only the fused dataset.")
    if dataset_key == "fused":
        # fused features are often strongest with these models
        models = {
            "Logistic Regression": LogisticRegression(
                C=4.0,
                max_iter=8000,
                solver="lbfgs",
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=1200,
                max_depth=None,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "SVM (RBF)": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                cache_size=1000,
                random_state=RANDOM_STATE,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                p=2,
            ),
        }
        if XGBClassifier is not None:
            kwargs = dict(
                n_estimators=700,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.90,
                colsample_bytree=0.90,
                min_child_weight=1,
                reg_lambda=1.5,
                reg_alpha=0.0,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
                objective="multi:softprob",
            )
            if use_gpu:
                kwargs["device"] = "cuda"
            models["XGBoost"] = XGBClassifier(**kwargs)
        if CatBoostClassifier is not None:
            kwargs = dict(
                iterations=1000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=5,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
            )
            if use_gpu:
                kwargs.update({"task_type": "GPU", "devices": "0"})
            models["CatBoost"] = CatBoostClassifier(**kwargs)
        return models
    raise AssertionError("Unreachable fused model configuration branch.")

def load_reduced_dataset_with_id(data_path: Path, min_samples=DEFAULT_MIN_SAMPLES, explicit_id_col=None):
    df = pd.read_excel(data_path)
    raw_row_count = int(len(df))
    if "target" not in df.columns:
        raise ValueError(f"Missing 'target' column in {data_path}. Found columns: {list(df.columns)}")

    id_col, candidate_df = detect_id_column(df, explicit_name=explicit_id_col)

    df.replace(["", " ", "NA", "NaN", "None", None, np.inf, -np.inf], np.nan, inplace=True)
    missing_target_rows = int(df["target"].isna().sum())
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(str).str.strip()
    vc = df["target"].value_counts()
    kept_families = vc[vc >= min_samples].index
    rare_family_rows = int((~df["target"].isin(kept_families)).sum())
    df = df[df["target"].isin(kept_families)].reset_index(drop=True)

    feature_exclude = ["target"]
    if id_col is not None:
        feature_exclude.append(id_col)

    feature_cols = [c for c in df.columns if c not in feature_exclude]
    # Preserve raw feature values. Imputation, categorical encoding, selection,
    # scaling, and SMOTE are fitted later using each training fold only.
    X = df[feature_cols].copy()
    y_raw = df["target"].astype(str).copy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    id_series = None
    if id_col is not None:
        id_series = df[id_col].astype(str).fillna("").str.strip()

    return {
        "source_path": str(data_path),
        "raw_row_count": raw_row_count,
        "rows_removed_by_class_filter": rare_family_rows,
        "rows_removed_missing_target": missing_target_rows,
        "preprocessing_audit": {
            "rows_before_preprocessing": raw_row_count,
            "rows_removed_missing_target": missing_target_rows,
            "rows_removed_rare_family": rare_family_rows,
            "rows_after_source_filtering": int(len(df)),
            "raw_feature_columns": list(map(str, feature_cols)),
            "numeric_features": [str(c) for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])],
            "categorical_features": [str(c) for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])],
            "missing_values_by_feature": {str(c): int(df[c].isna().sum()) for c in feature_cols},
            "fold_fitted_operations": [
                "numeric median imputation", "categorical constant imputation",
                "unknown-safe ordinal encoding", "mutual-information feature selection",
                "model-specific standardization", "SMOTE on training fold only",
            ],
        },
        "df": df,
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "class_names": class_names,
        "id_col": id_col,
        "id_series": id_series,
        "group_series": id_series,
        "candidate_id_columns": candidate_df,
    }

def derive_training_only_ensemble_weights(X_train: pd.DataFrame, y_train: np.ndarray, models: dict,
                                          alpha: float | None, smote_k: int, top_k: int = 3,
                                          inner_splits: int = 3):
    """Select models, alpha, and weights using only the current outer training fold."""
    counts = pd.Series(y_train).value_counts()
    effective_splits = min(int(inner_splits), int(counts.min())) if len(counts) else 0
    if effective_splits < 2:
        raise ValueError("At least two samples per class are required for training-only ensemble weighting.")
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_splits_list = list(splitter.split(X_train, y_train))
    all_classes = np.unique(y_train)
    score_rows = []
    oof_probabilities = {}
    for model_name, model in models.items():
        fold_scores = []
        model_oof = np.zeros((len(y_train), len(all_classes)), dtype=float)
        for inner_train_idx, inner_val_idx in inner_splits_list:
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_val = X_train.iloc[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val = y_train[inner_val_idx]
            X_res, y_res, X_val_eval = prepare_fold_data(
                X_inner_train, X_inner_val, y_inner_train,
                model_name=model_name, smote_k=smote_k,
            )
            estimator = safe_clone(model)
            estimator.fit(X_res, y_res)
            probability = align_proba_to_all_classes(
                estimator.predict_proba(X_val_eval), estimator.classes_, all_classes
            )
            model_oof[inner_val_idx] = probability
            prediction = all_classes[np.argmax(probability, axis=1)]
            fold_scores.append(f1_score(y_inner_val, prediction, average="weighted", zero_division=0))
        oof_probabilities[model_name] = model_oof
        score_rows.append({
            "Model": model_name,
            "Inner weighted F1": float(np.mean(fold_scores)),
            "Inner folds": effective_splits,
        })
    scores_df = pd.DataFrame(score_rows).sort_values("Inner weighted F1", ascending=False).reset_index(drop=True)
    selected = scores_df.head(min(int(top_k), len(scores_df))).copy()
    alpha_candidates = [float(alpha)] if alpha is not None else list(map(float, FUSED_ENSEMBLE_ALPHA_GRID))
    alpha_rows = []
    for candidate in alpha_candidates:
        candidate_raw = selected["Inner weighted F1"].clip(lower=1e-8) ** candidate
        candidate_weights = candidate_raw / candidate_raw.sum()
        probability = np.zeros((len(y_train), len(all_classes)), dtype=float)
        for model_name, weight in zip(selected["Model"], candidate_weights):
            probability += float(weight) * oof_probabilities[str(model_name)]
        prediction = all_classes[np.argmax(probability, axis=1)]
        alpha_rows.append({
            "Alpha": candidate,
            "Inner OOF weighted F1": float(f1_score(
                y_train, prediction, average="weighted", zero_division=0
            )),
            "Selection scope": "current outer training fold only",
        })
    alpha_scores_df = pd.DataFrame(alpha_rows).sort_values(
        ["Inner OOF weighted F1", "Alpha"], ascending=[False, True]
    ).reset_index(drop=True)
    selected_alpha = float(alpha_scores_df.iloc[0]["Alpha"])
    raw_weights = selected["Inner weighted F1"].clip(lower=1e-8) ** selected_alpha
    selected["Weight (normalized)"] = raw_weights / raw_weights.sum()
    weights = pd.Series(selected["Weight (normalized)"].values, index=selected["Model"].values, dtype=float)
    return weights, scores_df, selected_alpha, alpha_scores_df

def evaluate_dataset(
    dataset_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    class_names,
    output_dir: Path,
    models: dict,
    cv_folds=10,
    groups=None,
    validation_tag: str | None = None,
    ensemble_weight_power: float | None = None,
):
    plots_dir = ensure_dir(output_dir / "plots")
    all_classes = np.unique(y)
    splitter, split_name = build_cv_splitter(y, groups=groups, n_splits=cv_folds)
    print_header(f"VALIDATION PROTOCOL: {dataset_name}")
    print(f"Split strategy: {split_name}")
    if groups is not None:
        group_count = len(np.unique(pd.Series(groups).astype(str)))
        print(f"Grouping enabled with {group_count} unique groups")
    else:
        print("[WARN] No usable group labels were found; using stratified validation as a fallback.")

    summary = {}
    per_class_fold_metrics = {}
    ensemble_tracker = init_per_class_tracker(all_classes)
    top_confusions_frames = []
    smote_k = 3 if dataset_name.lower() == "disk" else 1
    split_iter = lambda: iter_splits(splitter, X, y, groups)

    fold_rows = []
    feature_selection_rows = []
    pooled_predictions = {}
    confusion_summary_rows = []
    top_confusion_frames = []

    for name, model in models.items():
        metrics = {"acc": [], "balanced_acc": [], "prec": [], "rec": [], "f1": [], "f1_macro": [], "auc": [], "auprc": [], "mcc": []}
        y_true_all_folds = []
        y_pred_all_folds = []
        y_prob_all_folds = []
        per_class_tracker = init_per_class_tracker(all_classes)

        print(f"\nTraining {dataset_name} -> {name} ...")
        try:
            for fold_no, (train_idx, test_idx) in enumerate(split_iter(), start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_res, y_res, X_test_eval, fold_artifacts = prepare_fold_data(
                    X_train, X_test, y_train, model_name=name, smote_k=smote_k,
                    return_artifacts=True,
                )
                selection_fold = fold_artifacts["selection_df"].copy()
                selection_fold.insert(0, "Fold", fold_no)
                selection_fold.insert(0, "Model", name)
                selection_fold.insert(0, "Dataset", dataset_name)
                feature_selection_rows.append(selection_fold)

                est = safe_clone(model)
                est.fit(X_res, y_res)
                y_pred = est.predict(X_test_eval)
                y_prob = est.predict_proba(X_test_eval)
                y_prob = align_proba_to_all_classes(y_prob, est.classes_, all_classes)

                acc, prec, rec, f1_w, f1_macro, auc_val, _ = compute_metrics(y_test, y_pred, y_prob, labels=all_classes)
                ext_metrics = compute_metrics_extended(y_test, y_pred, y_prob, labels=all_classes)
                update_per_class_tracker(per_class_tracker, y_test, y_pred, all_classes)

                for k, v in zip(["acc", "prec", "rec", "f1", "f1_macro", "auc"], [acc, prec, rec, f1_w, f1_macro, auc_val]):
                    metrics[k].append(v)
                metrics["auprc"].append(ext_metrics["auprc"])
                metrics["mcc"].append(ext_metrics["mcc"])
                metrics["balanced_acc"].append(ext_metrics["balanced_accuracy"])

                fold_rows.append({
                    "Dataset": dataset_name,
                    "Model": name,
                    "Fold": fold_no,
                    "Accuracy": acc,
                    "Balanced Accuracy": ext_metrics["balanced_accuracy"],
                    "Precision (weighted)": prec,
                    "Recall (weighted)": rec,
                    "F1 (weighted)": f1_w,
                    "F1 (macro)": f1_macro,
                    "AUC (OVR)": auc_val,
                    "AUPRC": ext_metrics["auprc"],
                    "MCC": ext_metrics["mcc"],
                })

                y_true_all_folds.append(y_test)
                y_pred_all_folds.append(y_pred)
                y_prob_all_folds.append(y_prob)

            y_true_all = np.concatenate(y_true_all_folds)
            y_pred_all = np.concatenate(y_pred_all_folds)
            y_prob_all = np.concatenate(y_prob_all_folds)

            plot_binary_or_multiclass_roc(
                y_true_all,
                y_prob_all,
                name,
                class_names,
                plots_dir / f"roc_{sanitize_filename(name)}.png",
            )

            cm = confusion_matrix(y_true_all, y_pred_all, labels=all_classes)
            plot_confusion_matrix(
                cm,
                class_names,
                plots_dir / f"confusion_matrix_{sanitize_filename(name)}.png",
                title=f"{dataset_name}: Confusion Matrix - {name}",
                normalize=False,
            )
            plot_confusion_matrix(
                cm,
                class_names,
                plots_dir / f"confusion_matrix_norm_{sanitize_filename(name)}.png",
                title=f"{dataset_name}: Normalized Confusion Matrix - {name}",
                normalize=True,
            )
            top_conf = build_top_confusions(cm, class_names, top_n=12)
            if not top_conf.empty:
                top_conf["Dataset"] = dataset_name
                top_conf["Model"] = name
                top_confusions_frames.append(top_conf)
                save_df(top_conf, output_dir / f"top_confusions_{sanitize_filename(name)}.csv", output_dir / f"top_confusions_{sanitize_filename(name)}.xlsx")

            acc_mean, acc_std, acc_lo, acc_hi = bootstrap_ci(metrics["acc"])
            bacc_mean, bacc_std, bacc_lo, bacc_hi = bootstrap_ci(metrics["balanced_acc"])
            prec_mean, prec_std, prec_lo, prec_hi = bootstrap_ci(metrics["prec"])
            rec_mean, rec_std, rec_lo, rec_hi = bootstrap_ci(metrics["rec"])
            f1_mean, f1_std, f1_lo, f1_hi = bootstrap_ci(metrics["f1"])
            f1m_mean, f1m_std, f1m_lo, f1m_hi = bootstrap_ci(metrics["f1_macro"])
            auc_mean, auc_std, auc_lo, auc_hi = bootstrap_ci(metrics["auc"])
            auprc_mean, auprc_std, auprc_lo, auprc_hi = bootstrap_ci(metrics["auprc"])
            mcc_mean, mcc_std, mcc_lo, mcc_hi = bootstrap_ci(metrics["mcc"])

            summary[name] = {
                "acc": acc_mean,
                "acc_std": acc_std,
                "acc_ci_low": acc_lo,
                "acc_ci_high": acc_hi,
                "balanced_acc": bacc_mean,
                "balanced_acc_std": bacc_std,
                "balanced_acc_ci_low": bacc_lo,
                "balanced_acc_ci_high": bacc_hi,
                "prec": prec_mean,
                "prec_std": prec_std,
                "prec_ci_low": prec_lo,
                "prec_ci_high": prec_hi,
                "rec": rec_mean,
                "rec_std": rec_std,
                "rec_ci_low": rec_lo,
                "rec_ci_high": rec_hi,
                "f1": f1_mean,
                "f1_std": f1_std,
                "f1_ci_low": f1_lo,
                "f1_ci_high": f1_hi,
                "f1_macro": f1m_mean,
                "f1_macro_std": f1m_std,
                "f1_macro_ci_low": f1m_lo,
                "f1_macro_ci_high": f1m_hi,
                "auc": auc_mean,
                "auc_std": auc_std,
                "auc_ci_low": auc_lo,
                "auc_ci_high": auc_hi,
                "auprc": auprc_mean,
                "auprc_std": auprc_std,
                "auprc_ci_low": auprc_lo,
                "auprc_ci_high": auprc_hi,
                "mcc": mcc_mean,
                "mcc_std": mcc_std,
                "mcc_ci_low": mcc_lo,
                "mcc_ci_high": mcc_hi,
            }
            pooled_predictions[name] = {
                "y_true": y_true_all,
                "y_pred": y_pred_all,
                "y_prob": y_prob_all,
                "cm": cm,
            }
            per_class_fold_metrics[name] = per_class_tracker
        except Exception as exc:
            print(f"[WARN] Skipping model '{name}' for {dataset_name}: {exc}")

    if not summary:
        raise RuntimeError(f"No model completed successfully for {dataset_name}.")

    results_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": vals.get("acc", np.nan),
            "Accuracy Std": vals.get("acc_std", np.nan),
            "Accuracy CI95 Low": vals.get("acc_ci_low", np.nan),
            "Accuracy CI95 High": vals.get("acc_ci_high", np.nan),
            "Balanced Accuracy": vals.get("balanced_acc", np.nan),
            "Balanced Accuracy Std": vals.get("balanced_acc_std", np.nan),
            "Balanced Accuracy CI95 Low": vals.get("balanced_acc_ci_low", np.nan),
            "Balanced Accuracy CI95 High": vals.get("balanced_acc_ci_high", np.nan),
            "Precision (weighted)": vals.get("prec", np.nan),
            "Precision Std": vals.get("prec_std", np.nan),
            "Precision CI95 Low": vals.get("prec_ci_low", np.nan),
            "Precision CI95 High": vals.get("prec_ci_high", np.nan),
            "Recall (weighted)": vals.get("rec", np.nan),
            "Recall Std": vals.get("rec_std", np.nan),
            "Recall CI95 Low": vals.get("rec_ci_low", np.nan),
            "Recall CI95 High": vals.get("rec_ci_high", np.nan),
            "F1 (weighted)": vals.get("f1", np.nan),
            "F1 Std": vals.get("f1_std", np.nan),
            "F1 CI95 Low": vals.get("f1_ci_low", np.nan),
            "F1 CI95 High": vals.get("f1_ci_high", np.nan),
            "F1 (macro)": vals.get("f1_macro", np.nan),
            "F1 Macro Std": vals.get("f1_macro_std", np.nan),
            "F1 Macro CI95 Low": vals.get("f1_macro_ci_low", np.nan),
            "F1 Macro CI95 High": vals.get("f1_macro_ci_high", np.nan),
            "AUC (OVR)": vals.get("auc", np.nan),
            "AUC Std": vals.get("auc_std", np.nan),
            "AUC CI95 Low": vals.get("auc_ci_low", np.nan),
            "AUC CI95 High": vals.get("auc_ci_high", np.nan),
            "AUPRC": vals.get("auprc", np.nan),
            "AUPRC Std": vals.get("auprc_std", np.nan),
            "AUPRC CI95 Low": vals.get("auprc_ci_low", np.nan),
            "AUPRC CI95 High": vals.get("auprc_ci_high", np.nan),
            "MCC": vals.get("mcc", np.nan),
            "MCC Std": vals.get("mcc_std", np.nan),
            "MCC CI95 Low": vals.get("mcc_ci_low", np.nan),
            "MCC CI95 High": vals.get("mcc_ci_high", np.nan),
            "Validation": split_name,
        }
        for name, vals in summary.items()
    ]).sort_values(["F1 (weighted)", "Accuracy"], ascending=False).reset_index(drop=True)
    save_df(results_df, output_dir / f"{dataset_name.lower()}_all_model_results.csv", output_dir / f"{dataset_name.lower()}_all_model_results.xlsx")

    class_acc_all = []
    class_metrics_all = []
    for model_name, metric_map in per_class_fold_metrics.items():
        class_acc_all.append(build_class_accuracy_df(metric_map, class_names, model_name=model_name, dataset_name=dataset_name))
        class_metrics_all.append(build_per_class_metrics_df(metric_map, class_names, model_name=model_name, dataset_name=dataset_name))
    class_acc_all_df = pd.concat(class_acc_all, ignore_index=True)
    class_metrics_all_df = pd.concat(class_metrics_all, ignore_index=True)
    save_df(class_acc_all_df, output_dir / f"{dataset_name.lower()}_per_model_class_accuracy.csv", output_dir / f"{dataset_name.lower()}_per_model_class_accuracy.xlsx")
    save_df(class_metrics_all_df, output_dir / f"{dataset_name.lower()}_per_model_per_family_metrics.csv",
            output_dir / f"{dataset_name.lower()}_per_model_per_family_metrics.xlsx")
    if feature_selection_rows:
        feature_selection_df = pd.concat(feature_selection_rows, ignore_index=True)
        save_df(feature_selection_df, output_dir / "fused_training_fold_feature_selection.csv",
                output_dir / "fused_training_fold_feature_selection.xlsx")
    else:
        feature_selection_df = pd.DataFrame()

    plot_metric_summary_bars(results_df, plots_dir / "metric_accuracy_summary.png", "Accuracy", "Accuracy Std", f"{dataset_name}: Accuracy with uncertainty")
    plot_metric_summary_bars(results_df, plots_dir / "metric_balanced_accuracy_summary.png", "Balanced Accuracy", "Balanced Accuracy Std", f"{dataset_name}: Balanced accuracy with uncertainty")
    plot_metric_summary_bars(results_df, plots_dir / "metric_f1w_summary.png", "F1 (weighted)", "F1 Std", f"{dataset_name}: Weighted F1 with uncertainty")
    plot_metric_summary_bars(results_df, plots_dir / "metric_f1macro_summary.png", "F1 (macro)", "F1 Macro Std", f"{dataset_name}: Macro F1 with uncertainty")
    plot_metric_summary_bars(results_df, plots_dir / "metric_auc_summary.png", "AUC (OVR)", "AUC Std", f"{dataset_name}: AUC with uncertainty")
    plot_fold_metric_boxplots(pd.DataFrame(fold_rows), plots_dir, dataset_name)

    ensemble_top_k = min(3, len(models))
    ensemble_weight_rows = []
    ensemble_alpha_rows = []
    ensemble_models_used = set()

    ens_metrics = {"acc": [], "balanced_acc": [], "prec_w": [], "rec_w": [], "f1_w": [], "f1_macro": [], "auc_ovr": [], "auprc": [], "mcc": []}
    ens_true_all = []
    ens_pred_all = []
    ens_prob_all = []
    for fold_no, (train_idx, test_idx) in enumerate(split_iter(), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        weights, inner_scores_df, selected_alpha, alpha_scores_df = derive_training_only_ensemble_weights(
            X_train, y_train, models, alpha=ensemble_weight_power,
            smote_k=smote_k, top_k=ensemble_top_k,
        )
        alpha_scores_df = alpha_scores_df.copy()
        alpha_scores_df.insert(0, "Outer Fold", fold_no)
        alpha_scores_df["Selected"] = alpha_scores_df["Alpha"] == selected_alpha
        alpha_scores_df["Requested policy"] = (
            "training-only inner-CV selection" if ensemble_weight_power is None
            else f"fixed alpha={float(ensemble_weight_power):g}"
        )
        ensemble_alpha_rows.append(alpha_scores_df)
        ensemble_models_used.update(map(str, weights.index))
        for model_name, inner_row in inner_scores_df.set_index("Model").iterrows():
            ensemble_weight_rows.append({
                "Outer Fold": fold_no,
                "Model": model_name,
                "Selected in Top-k": bool(model_name in weights.index),
                "Inner weighted F1": float(inner_row["Inner weighted F1"]),
                "Inner folds": int(inner_row["Inner folds"]),
                "Weight (normalized)": float(weights.get(model_name, 0.0)),
                "Weight %": float(100.0 * weights.get(model_name, 0.0)),
                "Top-k used": ensemble_top_k,
                "Weight power": selected_alpha,
                "Selection data": "current outer training fold only",
            })

        prob_ens = np.zeros((len(test_idx), len(all_classes)), dtype=float)
        for model_name in weights.index:
            X_res, y_res, X_test_eval = prepare_fold_data(
                X_train, X_test, y_train, model_name=model_name, smote_k=smote_k
            )
            est = safe_clone(models[model_name])
            est.fit(X_res, y_res)
            prob = est.predict_proba(X_test_eval)
            prob = align_proba_to_all_classes(prob, est.classes_, all_classes)
            prob_ens += float(weights[model_name]) * prob

        y_pred_ens = all_classes[np.argmax(prob_ens, axis=1)]
        acc, prec_w, rec_w, f1_w, f1_macro, auc_ovr = compute_metrics_ensemble(y_test, y_pred_ens, prob_ens, labels=all_classes)
        ext_ens = compute_metrics_extended(y_test, y_pred_ens, prob_ens, labels=all_classes)
        ens_metrics["acc"].append(acc)
        ens_metrics["prec_w"].append(prec_w)
        ens_metrics["rec_w"].append(rec_w)
        ens_metrics["f1_w"].append(f1_w)
        ens_metrics["f1_macro"].append(f1_macro)
        ens_metrics["auc_ovr"].append(auc_ovr)
        ens_metrics["auprc"].append(ext_ens["auprc"])
        ens_metrics["mcc"].append(ext_ens["mcc"])
        ens_metrics["balanced_acc"].append(ext_ens["balanced_accuracy"])

        fold_rows.append({
            "Dataset": dataset_name,
            "Model": f"Top-{ensemble_top_k} F1-weighted Soft Voting Ensemble",
            "Fold": fold_no,
            "Accuracy": acc,
            "Balanced Accuracy": ext_ens["balanced_accuracy"],
            "Precision (weighted)": prec_w,
            "Recall (weighted)": rec_w,
            "F1 (weighted)": f1_w,
            "F1 (macro)": f1_macro,
            "AUC (OVR)": auc_ovr,
        })

        ens_true_all.append(y_test)
        ens_pred_all.append(y_pred_ens)
        ens_prob_all.append(prob_ens)
        update_per_class_tracker(ensemble_tracker, y_test, y_pred_ens, all_classes)

    weights_df = pd.DataFrame(ensemble_weight_rows)
    save_df(weights_df, output_dir / f"{dataset_name.lower()}_ensemble_weights.csv",
            output_dir / f"{dataset_name.lower()}_ensemble_weights.xlsx")
    ensemble_alpha_df = pd.concat(ensemble_alpha_rows, ignore_index=True)
    save_df(ensemble_alpha_df, output_dir / f"{dataset_name.lower()}_ensemble_inner_alpha_selection.csv",
            output_dir / f"{dataset_name.lower()}_ensemble_inner_alpha_selection.xlsx")

    acc_mean, acc_std, acc_lo, acc_hi = bootstrap_ci(ens_metrics["acc"])
    bacc_mean, bacc_std, bacc_lo, bacc_hi = bootstrap_ci(ens_metrics["balanced_acc"])
    prec_mean, prec_std, prec_lo, prec_hi = bootstrap_ci(ens_metrics["prec_w"])
    rec_mean, rec_std, rec_lo, rec_hi = bootstrap_ci(ens_metrics["rec_w"])
    f1_mean, f1_std, f1_lo, f1_hi = bootstrap_ci(ens_metrics["f1_w"])
    f1m_mean, f1m_std, f1m_lo, f1m_hi = bootstrap_ci(ens_metrics["f1_macro"])
    auc_mean, auc_std, auc_lo, auc_hi = bootstrap_ci(ens_metrics["auc_ovr"])

    ensemble_row = pd.DataFrame([{
        "Model": f"Top-{ensemble_top_k} F1-weighted Soft Voting Ensemble",
        "Accuracy": acc_mean,
        "Accuracy Std": acc_std,
        "Accuracy CI95 Low": acc_lo,
        "Accuracy CI95 High": acc_hi,
        "Balanced Accuracy": bacc_mean,
        "Balanced Accuracy Std": bacc_std,
        "Balanced Accuracy CI95 Low": bacc_lo,
        "Balanced Accuracy CI95 High": bacc_hi,
        "Precision (weighted)": prec_mean,
        "Precision Std": prec_std,
        "Precision CI95 Low": prec_lo,
        "Precision CI95 High": prec_hi,
        "Recall (weighted)": rec_mean,
        "Recall Std": rec_std,
        "Recall CI95 Low": rec_lo,
        "Recall CI95 High": rec_hi,
        "F1 (weighted)": f1_mean,
        "F1 Std": f1_std,
        "F1 CI95 Low": f1_lo,
        "F1 CI95 High": f1_hi,
        "F1 (macro)": f1m_mean,
        "F1 Macro Std": f1m_std,
        "F1 Macro CI95 Low": f1m_lo,
        "F1 Macro CI95 High": f1m_hi,
        "AUC (OVR)": auc_mean,
        "AUC Std": auc_std,
        "AUC CI95 Low": auc_lo,
        "AUC CI95 High": auc_hi,
        "AUPRC": float(np.nanmean(ens_metrics["auprc"])) if len(ens_metrics["auprc"]) else np.nan,
        "MCC": float(np.nanmean(ens_metrics["mcc"])) if len(ens_metrics["mcc"]) else np.nan,
        "Validation": split_name,
    }])
    save_df(ensemble_row, output_dir / f"{dataset_name.lower()}_ensemble_metrics.csv", output_dir / f"{dataset_name.lower()}_ensemble_metrics.xlsx")

    ensemble_class_acc_df = build_class_accuracy_df(
        ensemble_tracker,
        class_names,
        model_name=f"Top-{ensemble_top_k} F1-weighted Soft Voting Ensemble",
        dataset_name=dataset_name,
    )
    save_df(
        ensemble_class_acc_df,
        output_dir / f"{dataset_name.lower()}_ensemble_class_accuracy.csv",
        output_dir / f"{dataset_name.lower()}_ensemble_class_accuracy.xlsx",
    )
    ensemble_per_family_df = build_per_class_metrics_df(
        ensemble_tracker, class_names,
        model_name=f"Top-{ensemble_top_k} F1-weighted Soft Voting Ensemble",
        dataset_name=dataset_name,
    )
    save_df(ensemble_per_family_df, output_dir / f"{dataset_name.lower()}_ensemble_per_family_metrics.csv",
            output_dir / f"{dataset_name.lower()}_ensemble_per_family_metrics.xlsx")

    ens_true_all = np.concatenate(ens_true_all)
    ens_pred_all = np.concatenate(ens_pred_all)
    ens_prob_all = np.concatenate(ens_prob_all)
    cm_ens = confusion_matrix(ens_true_all, ens_pred_all, labels=all_classes)
    plot_confusion_matrix(
        cm_ens,
        class_names,
        plots_dir / f"confusion_matrix_ensemble.png",
        title=f"{dataset_name}: Confusion Matrix - Top-{ensemble_top_k} Ensemble",
        normalize=False,
    )
    plot_confusion_matrix(
        cm_ens,
        class_names,
        plots_dir / f"confusion_matrix_norm_ensemble.png",
        title=f"{dataset_name}: Normalized Confusion Matrix - Top-{ensemble_top_k} Ensemble",
        normalize=True,
    )
    plot_binary_or_multiclass_roc(
        ens_true_all,
        ens_prob_all,
        f"Top-{ensemble_top_k} Ensemble",
        class_names,
        plots_dir / f"roc_ensemble.png",
    )

    ensemble_class_acc_df["Dataset"] = dataset_name
    # Optional combined confusion summary across all models in this dataset.
    if top_confusions_frames:
        top_confusions_all_df = pd.concat(top_confusions_frames, ignore_index=True)
        save_df(
            top_confusions_all_df,
            output_dir / f"{dataset_name.lower()}_top_confusions_all.csv",
            output_dir / f"{dataset_name.lower()}_top_confusions_all.xlsx",
        )

    fold_metrics_df = pd.DataFrame(fold_rows)
    save_df(fold_metrics_df, output_dir / f"{dataset_name.lower()}_fold_metrics.csv", output_dir / f"{dataset_name.lower()}_fold_metrics.xlsx")

    validation_info = {
        "dataset": dataset_name,
        "validation_protocol": split_name,
        "grouped_validation": groups is not None,
        "n_splits": cv_folds,
        "ensemble_top_k": ensemble_top_k,
        "ensemble_models": sorted(ensemble_models_used),
        "ensemble_weight_selection": "inner stratified CV inside each outer training fold",
        "ensemble_alpha_policy": (
            "selected independently by inner CV in each outer training fold"
            if ensemble_weight_power is None else f"pre-specified fixed alpha={float(ensemble_weight_power):g}"
        ),
        "note": validation_tag or "Grouped validation uses sample-id-derived groups when available; fused rows use fusion identifiers.",
    }
    Path(output_dir / "validation_protocol.json").write_text(json.dumps(validation_info, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "results_df": results_df,
        "per_model_class_accuracy_df": class_acc_all_df,
        "per_model_per_family_metrics_df": class_metrics_all_df,
        "ensemble_class_accuracy_df": ensemble_class_acc_df,
        "ensemble_per_family_metrics_df": ensemble_per_family_df,
        "ensemble_metrics_df": ensemble_row,
        "weights_df": weights_df,
        "ensemble_alpha_selection_df": ensemble_alpha_df,
        "validation_info": validation_info,
        "fold_metrics_df": fold_metrics_df,
        "pooled_predictions": pooled_predictions,
        "per_class_fold_metrics": per_class_fold_metrics,
        "ensemble_per_class_fold_metrics": ensemble_tracker,
        "feature_selection_df": feature_selection_df,
    }

def rename_feature_block(X: pd.DataFrame, prefix: str):
    return X.rename(columns={c: f"{prefix}__{c}" for c in X.columns})

def normalize_sample_identifier(value) -> str | None:
    """Canonicalize a sample identifier without inventing or truncating it."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "na", "n/a"}:
        return None
    text = re.sub(r"^(sha[-_ ]?256|hash)\s*[:=]\s*", "", text)
    text = re.sub(r"\s+", "", text)
    return text or None

def _audited_modality_rows(bundle: dict, modality: str):
    ids = pd.Series(bundle["id_series"], index=bundle["df"].index).map(normalize_sample_identifier)
    targets = bundle["df"]["target"].astype(str).str.strip()
    work = pd.DataFrame({"normalized_id": ids, "target": targets}, index=bundle["df"].index)
    work = work[work["normalized_id"].notna()].copy()
    work["source_row_index"] = work.index.astype(int)

    target_counts = work.groupby("normalized_id")["target"].nunique()
    conflicting_ids = set(target_counts[target_counts > 1].index.astype(str))
    valid = work[~work["normalized_id"].isin(conflicting_ids)].copy()
    duplicate_mask = valid.duplicated("normalized_id", keep=False)
    deduplicated = valid.drop_duplicates("normalized_id", keep="first").copy()

    sha256_mask = deduplicated["normalized_id"].astype(str).str.fullmatch(r"[a-f0-9]{64}")
    audit = {
        "Scope": modality,
        "Source file": str(bundle.get("source_path", "")),
        "Detected ID column": str(bundle.get("id_col", "")),
        "Rows before preprocessing": int(bundle.get("raw_row_count", len(bundle["df"]))),
        "Rows removed for missing target": int(bundle.get("rows_removed_missing_target", 0)),
        "Rows removed by class filter": int(bundle.get("rows_removed_by_class_filter", 0)),
        "Rows after source preprocessing": int(len(bundle["df"])),
        "Rows with missing/invalid ID": int(ids.isna().sum()),
        "Unique usable IDs": int(deduplicated["normalized_id"].nunique()),
        "Duplicate rows beyond first": int(len(valid) - len(deduplicated)),
        "IDs appearing more than once": int(valid.loc[duplicate_mask, "normalized_id"].nunique()),
        "IDs with conflicting labels": int(len(conflicting_ids)),
        "Usable IDs that are SHA-256": int(sha256_mask.sum()),
        "SHA-256 proportion": float(sha256_mask.mean()) if len(sha256_mask) else 0.0,
    }
    conflicts = work[work["normalized_id"].isin(conflicting_ids)].copy()
    if not conflicts.empty:
        conflicts.insert(0, "modality", modality)
    return deduplicated, audit, conflicts

def build_fused_by_id(bundles: dict):
    """Build a defensible fused benchmark from exact normalized identifiers only."""
    for key in FUSION_MODALITIES:
        bundle = bundles[key]
        if bundle["id_col"] is None or bundle["id_series"] is None:
            raise ValueError(
                f"{key} has no usable sample identifier. Exact fused construction requires an ID "
                f"column in every modality; pass --{key}-id-column if automatic detection missed it."
            )

    audited = {}
    audit_rows = []
    conflict_frames = []
    for key in FUSION_MODALITIES:
        deduplicated, audit, conflicts = _audited_modality_rows(bundles[key], key)
        audited[key] = deduplicated
        audit_rows.append(audit)
        if not conflicts.empty:
            conflict_frames.append(conflicts)

    shared_ids = set.intersection(*(set(audited[key]["normalized_id"]) for key in FUSION_MODALITIES))
    if not shared_ids:
        raise ValueError(
            "No exact normalized sample identifiers are shared by disk, memory, and network. "
            "The code will not silently replace sample-level alignment with target-label pseudo-fusion."
        )

    label_table = None
    for key in FUSION_MODALITIES:
        block = audited[key][["normalized_id", "target"]].rename(columns={"target": f"target_{key}"})
        label_table = block if label_table is None else label_table.merge(block, on="normalized_id", how="inner")
    label_table = label_table[label_table["normalized_id"].isin(shared_ids)].copy()
    same_label = label_table[[f"target_{key}" for key in FUSION_MODALITIES]].nunique(axis=1) == 1
    cross_modality_conflicts = label_table.loc[~same_label].copy()
    valid_label_table = label_table.loc[same_label].copy()
    valid_ids = sorted(valid_label_table["normalized_id"].astype(str))
    if not valid_ids:
        raise ValueError("All exactly matched identifiers have conflicting family labels across modalities.")
    final_family_count = int(valid_label_table["target_disk"].nunique())

    target_ref = valid_label_table[["normalized_id", "target_disk"]].rename(
        columns={"normalized_id": "fusion_id", "target_disk": "target"}
    )
    fused = target_ref.copy()
    for key in FUSION_MODALITIES:
        retained_rows = audited[key].set_index("normalized_id").loc[valid_ids]
        X_block = bundles[key]["X"].loc[retained_rows["source_row_index"].astype(int).tolist()].copy()
        X_block.index = valid_ids
        X_block = rename_feature_block(X_block, key)
        X_block.insert(0, "fusion_id", valid_ids)
        fused = fused.merge(X_block, on="fusion_id", how="inner", validate="one_to_one")

    for row in audit_rows:
        modality = row["Scope"]
        ids = set(audited[modality]["normalized_id"])
        row["IDs in exact three-way intersection before label check"] = int(len(ids & shared_ids))
        row["Unmatched usable IDs"] = int(len(ids - shared_ids))
        row["Final aligned IDs"] = int(len(valid_ids))
        row["Final aligned families"] = final_family_count
    audit_rows.append({
        "Scope": "three_way_alignment",
        "Source file": "",
        "Detected ID column": "",
        "Rows before preprocessing": int(sum(bundles[k].get("raw_row_count", len(bundles[k]["df"])) for k in FUSION_MODALITIES)),
        "Rows removed for missing target": int(sum(bundles[k].get("rows_removed_missing_target", 0) for k in FUSION_MODALITIES)),
        "Rows removed by class filter": int(sum(bundles[k].get("rows_removed_by_class_filter", 0) for k in FUSION_MODALITIES)),
        "Rows after source preprocessing": int(sum(len(bundles[k]["df"]) for k in FUSION_MODALITIES)),
        "Rows with missing/invalid ID": int(sum(r["Rows with missing/invalid ID"] for r in audit_rows)),
        "Unique usable IDs": int(len(set.union(*(set(audited[k]["normalized_id"]) for k in FUSION_MODALITIES)))),
        "Duplicate rows beyond first": int(sum(r["Duplicate rows beyond first"] for r in audit_rows)),
        "IDs appearing more than once": int(sum(r["IDs appearing more than once"] for r in audit_rows)),
        "IDs with conflicting labels": int(sum(r["IDs with conflicting labels"] for r in audit_rows)),
        "Usable IDs that are SHA-256": int(sum(r["Usable IDs that are SHA-256"] for r in audit_rows)),
        "SHA-256 proportion": float(np.mean([r["SHA-256 proportion"] for r in audit_rows])),
        "IDs in exact three-way intersection before label check": int(len(shared_ids)),
        "Unmatched usable IDs": np.nan,
        "Final aligned IDs": int(len(valid_ids)),
        "Final aligned families": final_family_count,
        "Cross-modality label conflicts": int(len(cross_modality_conflicts)),
    })

    conflicts = pd.concat(conflict_frames, ignore_index=True) if conflict_frames else pd.DataFrame()
    if not cross_modality_conflicts.empty:
        cross_modality_conflicts = cross_modality_conflicts.copy()
        cross_modality_conflicts.insert(0, "modality", "cross_modality")
        conflicts = pd.concat([conflicts, cross_modality_conflicts], ignore_index=True, sort=False)

    fused = fused.sort_values("fusion_id").reset_index(drop=True)
    return fused, "id_exact", pd.DataFrame(audit_rows), conflicts

def build_fused_by_target_pairing(bundles: dict, random_state=RANDOM_STATE):
    print("[WARN] Using target-based pseudo-pairing fusion. This is not true sample-level fusion.")
    rng = np.random.default_rng(random_state)

    shared_targets = None
    for key, bundle in bundles.items():
        targets = set(bundle["df"]["target"].astype(str).tolist())
        shared_targets = targets if shared_targets is None else shared_targets & targets

    if not shared_targets:
        raise ValueError("No common target labels across datasets for target-based pairing.")

    shared_targets = sorted(shared_targets)
    paired_parts = []

    for key, bundle in bundles.items():
        df_block = bundle["df"].copy()
        X_block = bundle["X"].copy()
        tmp = pd.concat([df_block[["target"]].reset_index(drop=True), X_block.reset_index(drop=True)], axis=1)
        tmp = tmp[tmp["target"].astype(str).isin(shared_targets)].copy()

        class_counts = tmp["target"].astype(str).value_counts().to_dict()
        paired_rows = []
        for target in shared_targets:
            class_df = tmp[tmp["target"].astype(str) == str(target)].copy().reset_index(drop=True)
            idx = np.arange(len(class_df))
            rng.shuffle(idx)
            class_df = class_df.iloc[idx].reset_index(drop=True)
            class_df["pair_idx"] = np.arange(len(class_df))
            paired_rows.append(class_df)

        paired = pd.concat(paired_rows, ignore_index=True)
        paired_parts.append((key, paired))

    # min count per class across modalities
    min_class_counts = {}
    for target in shared_targets:
        counts = []
        for _, paired in paired_parts:
            counts.append(int((paired["target"].astype(str) == str(target)).sum()))
        min_class_counts[str(target)] = min(counts)

    trimmed_parts = []
    for key, paired in paired_parts:
        keep_rows = []
        for target in shared_targets:
            class_df = paired[paired["target"].astype(str) == str(target)].copy()
            m = min_class_counts[str(target)]
            class_df = class_df.iloc[:m].copy()
            keep_rows.append(class_df)
        trimmed = pd.concat(keep_rows, ignore_index=True)
        trimmed_parts.append((key, trimmed))

    fused = None
    for key, trimmed in trimmed_parts:
        block = trimmed.copy()
        # pair_idx is an alignment bookkeeping field, never a predictive feature.
        features = block.drop(columns=["target", "pair_idx"]).copy()
        features = rename_feature_block(features, key)
        features["target"] = block["target"].astype(str).values
        features["pair_idx"] = block["pair_idx"].astype(int).values

        if fused is None:
            fused = features.copy()
        else:
            fused = fused.merge(features, on=["target", "pair_idx"], how="inner")

    fused["fusion_id"] = fused["target"].astype(str) + "__" + fused["pair_idx"].astype(str)
    cols = ["fusion_id", "target"] + [c for c in fused.columns if c not in ["fusion_id", "target", "pair_idx"]] + ["pair_idx"]
    fused = fused[cols].copy()
    return fused, "target_pseudo_pairing"

def construct_fused_dataset(bundles: dict, fusion_mode: str, allow_pseudo_fusion: bool):
    """Construct exact fusion when possible, otherwise an explicitly separated auto sensitivity cohort."""
    empty = pd.DataFrame()
    if fusion_mode == "target":
        if not allow_pseudo_fusion:
            raise ValueError(
                "--fusion-mode target is a pseudo-fusion sensitivity analysis, not sample-level fusion. "
                "Re-run with --allow-pseudo-fusion only if that limitation is intentional."
            )
        fused, method = build_fused_by_target_pairing(bundles)
        limitation = {
            "exact_sample_alignment_available": False,
            "requested_mode": "target",
            "executed_mode": method,
            "reason": "Target-label pseudo-pairing was explicitly requested by the user.",
            "reporting_rule": "Sensitivity analysis only; do not call this genuine sample-level multimodal fusion.",
        }
        return fused, method, empty, empty, limitation

    if fusion_mode == "id":
        fused, method, audit, conflicts = build_fused_by_id(bundles)
        return fused, method, audit, conflicts, None

    if fusion_mode != "auto":
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    missing_modalities = [
        key for key in FUSION_MODALITIES
        if bundles[key].get("id_col") is None or bundles[key].get("id_series") is None
    ]
    exact_failure = None
    if not missing_modalities:
        try:
            fused, method, audit, conflicts = build_fused_by_id(bundles)
            return fused, method, audit, conflicts, None
        except ValueError as exc:
            exact_failure = str(exc)
    else:
        exact_failure = "Missing usable identifier columns in: " + ", ".join(missing_modalities)

    print_header("EXACT SAMPLE ALIGNMENT UNAVAILABLE")
    print(f"[LIMITATION] {exact_failure}")
    print("[LIMITATION] Continuing only with a separately stored target-label pseudo-fusion sensitivity analysis.")
    print("[LIMITATION] These results must not be reported as genuine sample-level multimodal fusion.")
    fused, method = build_fused_by_target_pairing(bundles)
    limitation = {
        "exact_sample_alignment_available": False,
        "requested_mode": "auto",
        "executed_mode": method,
        "reason": exact_failure,
        "modalities_without_detected_identifiers": missing_modalities,
        "detected_identifier_columns": {
            key: bundles[key].get("id_col") for key in FUSION_MODALITIES
        },
        "candidate_identifier_evidence_files": {
            key: f"construction/{key}_candidate_id_columns.csv" for key in FUSION_MODALITIES
        },
        "reporting_rule": "Sensitivity analysis only; do not call this genuine sample-level multimodal fusion.",
        "remediation": (
            "Add the same exact sample identifier (preferably SHA-256) to all three source files, "
            "or pass the correct --disk-id-column, --memory-id-column, and --network-id-column names."
        ),
    }
    return fused, method, empty, empty, limitation

def prepare_fused_xy(fused_df: pd.DataFrame):
    y_raw = fused_df["target"].astype(str).copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    drop_cols = [c for c in ["fusion_id", "target", "pair_idx"] if c in fused_df.columns]
    X = fused_df.drop(columns=drop_cols).copy().replace([np.inf, -np.inf], np.nan)
    return X, y, y_raw, class_names

def save_fused_feature_manifest(X_fused: pd.DataFrame, fused_df: pd.DataFrame, output_dir: Path,
                                rationale_json: str | None = None):
    rationale = {}
    if rationale_json:
        with open(rationale_json, "r", encoding="utf-8") as handle:
            rationale = json.load(handle)
        if not isinstance(rationale, dict):
            raise ValueError("--fused-feature-rationale-json must contain a JSON object mapping feature names to rationales.")

    rows = []
    for feature in X_fused.columns.astype(str):
        modality, source_feature = feature.split("__", 1) if "__" in feature else ("unassigned", feature)
        raw_series = X_fused[feature]
        is_numeric = pd.api.types.is_numeric_dtype(raw_series)
        numeric_series = pd.to_numeric(raw_series, errors="coerce") if is_numeric else None
        reason = rationale.get(feature, rationale.get(source_feature, ""))
        rows.append({
            "Fused feature": feature,
            "Evidence stream": modality,
            "Source feature": source_feature,
            "Raw dtype": str(raw_series.dtype),
            "Feature type": "numeric" if is_numeric else "categorical",
            "Missing before fold preprocessing": int(raw_series.isna().sum()),
            "Unique values": int(raw_series.nunique(dropna=True)),
            "Zero proportion": float((numeric_series.fillna(0) == 0).mean()) if is_numeric else np.nan,
            "Fold-fitted transformation": (
                "training-median imputation" if is_numeric
                else "training-fold missing-category imputation and unknown-safe ordinal encoding"
            ),
            "Selection provenance": "At most eight features per stream selected by mutual information fitted on each training fold; all supplied features retained when the stream contains eight or fewer",
            "Scientific rationale": reason if reason else "NOT SUPPLIED - must be justified from the study protocol/literature",
        })
    manifest = pd.DataFrame(rows)
    save_df(manifest, output_dir / "fused_feature_manifest.csv", output_dir / "fused_feature_manifest.xlsx")

    modality_summary = manifest.groupby("Evidence stream", as_index=False).agg(
        **{"Feature count": ("Fused feature", "count"),
           "Features without supplied rationale": ("Scientific rationale", lambda s: int(s.astype(str).str.startswith("NOT SUPPLIED").sum()))}
    )
    save_df(modality_summary, output_dir / "fused_feature_count_by_stream.csv",
            output_dir / "fused_feature_count_by_stream.xlsx")
    return manifest

def save_fused_class_balance(y_raw_fused: pd.Series, output_dir: Path):
    counts = pd.Series(y_raw_fused).astype(str).value_counts().rename_axis("Family").reset_index(name="Samples")
    total = int(counts["Samples"].sum())
    counts["Proportion"] = counts["Samples"] / max(1, total)
    max_count = int(counts["Samples"].max()) if not counts.empty else 0
    min_count = int(counts["Samples"].min()) if not counts.empty else 0
    counts["Relative to largest class"] = counts["Samples"] / max(1, max_count)
    save_df(counts, output_dir / "fused_class_balance.csv", output_dir / "fused_class_balance.xlsx")
    report = {
        "dataset": "Fused",
        "samples": total,
        "families": int(len(counts)),
        "largest_class_samples": max_count,
        "smallest_class_samples": min_count,
        "imbalance_ratio_largest_to_smallest": float(max_count / min_count) if min_count else None,
        "training_only_treatment": "SMOTE is fitted separately inside each outer training fold; test folds are never resampled.",
    }
    (output_dir / "fused_class_balance_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return counts, report

def run_fused_ensemble_diagnostics(fused_output: dict, class_names, output_dir: Path,
                                   alpha_grid=FUSED_ENSEMBLE_ALPHA_GRID, top_k: int = 3):
    predictions = fused_output["pooled_predictions"]
    results_df = fused_output["results_df"].copy()
    ranked = results_df[results_df["Model"].isin(predictions)].sort_values(
        ["F1 (weighted)", "Accuracy"], ascending=False
    )
    model_names = ranked["Model"].head(min(top_k, len(ranked))).astype(str).tolist()
    if not model_names:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    y_true = np.asarray(predictions[model_names[0]]["y_true"]).reshape(-1)
    all_classes = np.arange(len(class_names))
    for model_name in model_names[1:]:
        candidate_y_true = np.asarray(predictions[model_name]["y_true"]).reshape(-1)
        if not np.array_equal(y_true, candidate_y_true):
            raise RuntimeError("Pooled fused predictions are not aligned across models; ensemble diagnostics aborted.")

    f1_map = ranked.set_index("Model")["F1 (weighted)"].astype(float)
    sensitivity_rows = []
    for alpha in alpha_grid:
        raw = f1_map.loc[model_names].clip(lower=1e-8) ** float(alpha)
        weights = raw / raw.sum()
        probability = np.zeros_like(np.asarray(predictions[model_names[0]]["y_prob"]), dtype=float)
        for model_name in model_names:
            probability += float(weights[model_name]) * np.asarray(predictions[model_name]["y_prob"], dtype=float)
        y_pred = all_classes[np.argmax(probability, axis=1)]
        ext = compute_metrics_extended(y_true, y_pred, probability, labels=all_classes)
        sensitivity_rows.append({
            "Configuration": f"Soft voting alpha={float(alpha):g}",
            "Alpha": float(alpha),
            "Weighting": "equal" if float(alpha) == 0.0 else "F1 raised to alpha",
            "Models": "; ".join(model_names),
            "Weights": "; ".join(f"{m}={weights[m]:.8f}" for m in model_names),
            "Role": "Post-hoc OOF sensitivity only; alpha must be fixed before final evaluation",
            **ext,
        })
    best_single_name = model_names[0]
    best_single = predictions[best_single_name]
    best_single_metrics = compute_metrics_extended(
        y_true, np.asarray(best_single["y_pred"]).reshape(-1),
        np.asarray(best_single["y_prob"]), labels=all_classes
    )
    sensitivity_rows.append({
        "Configuration": "Best single-model baseline",
        "Alpha": np.nan,
        "Weighting": "not applicable",
        "Models": best_single_name,
        "Weights": f"{best_single_name}=1.0",
        "Role": "Baseline",
        **best_single_metrics,
    })
    sensitivity_df = pd.DataFrame(sensitivity_rows).reset_index(drop=True)
    save_df(sensitivity_df, output_dir / "fused_ensemble_alpha_sensitivity.csv",
            output_dir / "fused_ensemble_alpha_sensitivity.xlsx")

    flattened_predictions = {
        name: np.asarray(predictions[name]["y_pred"]).reshape(-1) for name in model_names
    }
    for name, values in flattened_predictions.items():
        if len(values) != len(y_true):
            raise RuntimeError(
                f"Prediction length mismatch for {name}: {len(values)} predictions for {len(y_true)} labels."
            )
    # Rows are samples and columns are models. Explicit rowvar=False prevents
    # accidental sample-by-sample correlations when a predictor returns (n, 1).
    error_matrix = np.column_stack([
        (flattened_predictions[name] != y_true).astype(float) for name in model_names
    ])
    if len(model_names) == 1:
        corr = np.ones((1, 1), dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(error_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=model_names, columns=model_names)
    corr_df.index.name = "Model"
    corr_df.to_csv(output_dir / "fused_model_error_correlation.csv")
    corr_df.to_excel(output_dir / "fused_model_error_correlation.xlsx")

    diversity_rows = []
    for i, left in enumerate(model_names):
        for right in model_names[i + 1:]:
            pred_left = flattened_predictions[left]
            pred_right = flattened_predictions[right]
            err_left = pred_left != y_true
            err_right = pred_right != y_true
            diversity_rows.append({
                "Model A": left,
                "Model B": right,
                "Prediction disagreement": float(np.mean(pred_left != pred_right)),
                "Double-fault rate": float(np.mean(err_left & err_right)),
                "At least one model correct": float(np.mean(~(err_left & err_right))),
            })
    diversity_df = pd.DataFrame(diversity_rows)
    save_df(diversity_df, output_dir / "fused_model_diversity.csv", output_dir / "fused_model_diversity.xlsx")
    return sensitivity_df, corr_df, diversity_df

def run_fused_imbalance_comparison(X: pd.DataFrame, y: np.ndarray, class_names, groups, model_name: str,
                                   model, output_dir: Path, cv_folds: int):
    splitter, split_name = build_cv_splitter(y, groups=groups, n_splits=cv_folds)
    all_classes = np.unique(y)
    overall_rows = []
    class_rows = []
    for strategy, apply_smote in [("No resampling", False), ("Training-fold SMOTE", True)]:
        fold_metrics = []
        tracker = init_per_class_tracker(all_classes)
        for fold_no, (train_idx, test_idx) in enumerate(iter_splits(splitter, X, y, groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_res, y_res, X_test_eval = prepare_fold_data(
                X_train, X_test, y_train, model_name=model_name, smote_k=1, apply_smote=apply_smote
            )
            est = safe_clone(model)
            est.fit(X_res, y_res)
            y_pred = est.predict(X_test_eval)
            y_prob = align_proba_to_all_classes(est.predict_proba(X_test_eval), est.classes_, all_classes)
            ext = compute_metrics_extended(y_test, y_pred, y_prob, labels=all_classes)
            ext.update({"Strategy": strategy, "Fold": fold_no})
            fold_metrics.append(ext)
            update_per_class_tracker(tracker, y_test, y_pred, all_classes)
        fold_df = pd.DataFrame(fold_metrics)
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc"]:
            mean, std, low, high = bootstrap_ci(fold_df[metric])
            overall_rows.append({
                "Model": model_name, "Validation": split_name, "Strategy": strategy, "Metric": metric,
                "Mean": mean, "Std": std, "CI95 Low": low, "CI95 High": high,
            })
        cls_df = build_per_class_metrics_df(tracker, class_names, model_name, "Fused")
        cls_df.insert(2, "Imbalance strategy", strategy)
        class_rows.append(cls_df)
    overall_df = pd.DataFrame(overall_rows)
    class_df = pd.concat(class_rows, ignore_index=True)
    save_df(overall_df, output_dir / "fused_imbalance_strategy_comparison.csv",
            output_dir / "fused_imbalance_strategy_comparison.xlsx")
    save_df(class_df, output_dir / "fused_imbalance_strategy_per_family.csv",
            output_dir / "fused_imbalance_strategy_per_family.xlsx")
    return overall_df, class_df

def run_fused_temporal_evaluation(X: pd.DataFrame, y: np.ndarray, class_names, timestamps,
                                  model_name: str, model, output_dir: Path,
                                  train_fraction: float = 0.7):
    parsed_time = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    valid = parsed_time.notna().to_numpy()
    status = {
        "requested": True,
        "valid_timestamp_rows": int(valid.sum()),
        "invalid_timestamp_rows": int((~valid).sum()),
        "train_fraction": float(train_fraction),
    }
    if valid.sum() < 2:
        status.update({"completed": False, "reason": "Fewer than two valid timestamps."})
        (output_dir / "fused_temporal_evaluation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return None
    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(parsed_time.iloc[valid_indices].astype("int64").to_numpy())]
    cut = int(np.floor(len(order) * float(train_fraction)))
    cut = max(1, min(cut, len(order) - 1))
    train_idx, test_idx = order[:cut], order[cut:]
    y_train, y_test = y[train_idx], y[test_idx]
    missing_train = sorted(set(np.unique(y)) - set(np.unique(y_train)))
    if missing_train:
        status.update({
            "completed": False,
            "reason": "Chronological training partition does not contain every family.",
            "missing_training_class_indices": list(map(int, missing_train)),
        })
        (output_dir / "fused_temporal_evaluation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return None

    X_res, y_res, X_test_eval = prepare_fold_data(
        X.iloc[train_idx], X.iloc[test_idx], y_train,
        model_name=model_name, smote_k=1,
    )
    estimator = safe_clone(model)
    estimator.fit(X_res, y_res)
    y_pred = estimator.predict(X_test_eval)
    probability = align_proba_to_all_classes(
        estimator.predict_proba(X_test_eval), estimator.classes_, np.unique(y)
    )
    metrics = compute_metrics_extended(y_test, y_pred, probability, labels=np.unique(y))
    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "Train samples": int(len(train_idx)),
        "Test samples": int(len(test_idx)),
        "Train start": str(parsed_time.iloc[train_idx].min()),
        "Train end": str(parsed_time.iloc[train_idx].max()),
        "Test start": str(parsed_time.iloc[test_idx].min()),
        "Test end": str(parsed_time.iloc[test_idx].max()),
        **metrics,
    }])
    tracker = init_per_class_tracker(np.unique(y))
    update_per_class_tracker(tracker, y_test, y_pred, np.unique(y))
    family_df = build_per_class_metrics_df(tracker, class_names, model_name, "Fused chronological holdout")
    save_df(metrics_df, output_dir / "fused_temporal_evaluation_metrics.csv",
            output_dir / "fused_temporal_evaluation_metrics.xlsx")
    save_df(family_df, output_dir / "fused_temporal_evaluation_per_family.csv",
            output_dir / "fused_temporal_evaluation_per_family.xlsx")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plot_confusion_matrix(cm, class_names, ensure_dir(output_dir / "plots") / "fused_temporal_confusion_matrix.png",
                          "Fused chronological holdout confusion matrix", normalize=False)
    status.update({"completed": True, "reason": None})
    (output_dir / "fused_temporal_evaluation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return {"metrics_df": metrics_df, "per_family_df": family_df, "status": status}

def run_fused_modality_ablation(X: pd.DataFrame, y: np.ndarray, class_names, groups, selected_model_name: str,
                                selected_model, full_fused_output: dict, output_dir: Path, cv_folds: int,
                                ensemble_weight_power: float | None):
    specifications = [
        ("Disk", ("disk",)),
        ("Memory", ("memory",)),
        ("Network", ("network",)),
        ("Disk+Memory", ("disk", "memory")),
        ("Disk+Network", ("disk", "network")),
        ("Memory+Network", ("memory", "network")),
        ("Disk+Memory+Network", FUSION_MODALITIES),
    ]
    ablation_root = ensure_dir(output_dir / "fused_modality_ablation")
    class_frames = []
    family_metric_frames = []
    summary_rows = []
    combination_results = {}
    for label, modalities in specifications:
        columns = [c for c in X.columns if c.split("__", 1)[0] in modalities]
        if not columns:
            raise ValueError(f"No fused feature columns found for ablation combination {label}.")
        if len(modalities) == len(FUSION_MODALITIES):
            result = full_fused_output
        else:
            result = evaluate_dataset(
                dataset_name=f"Fused aligned {label}",
                X=X[columns], y=y, class_names=class_names,
                output_dir=ensure_dir(ablation_root / sanitize_filename(label.lower())),
                models={selected_model_name: selected_model}, cv_folds=cv_folds, groups=groups,
                validation_tag="same exact-ID-aligned cohort",
                ensemble_weight_power=ensemble_weight_power,
            )
        result_row = result["results_df"]
        combination_results[label] = result
        result_row = result_row[result_row["Model"] == selected_model_name].iloc[0]
        summary_rows.append({
            "Combination": label,
            "Modalities": "; ".join(modalities),
            "Features": int(len(columns)),
            "Model": selected_model_name,
            "Accuracy": float(result_row["Accuracy"]),
            "Accuracy CI95 Low": float(result_row["Accuracy CI95 Low"]),
            "Accuracy CI95 High": float(result_row["Accuracy CI95 High"]),
            "Balanced Accuracy": float(result_row["Balanced Accuracy"]),
            "Balanced Accuracy CI95 Low": float(result_row["Balanced Accuracy CI95 Low"]),
            "Balanced Accuracy CI95 High": float(result_row["Balanced Accuracy CI95 High"]),
            "F1 (macro)": float(result_row["F1 (macro)"]),
            "F1 Macro CI95 Low": float(result_row["F1 Macro CI95 Low"]),
            "F1 Macro CI95 High": float(result_row["F1 Macro CI95 High"]),
        })
        cls = result["per_model_class_accuracy_df"]
        cls = cls[cls["Model"] == selected_model_name].copy()
        cls["Combination"] = label
        class_frames.append(cls)
        family_metrics = result["per_model_per_family_metrics_df"]
        family_metrics = family_metrics[family_metrics["Model"] == selected_model_name].copy()
        family_metrics["Combination"] = label
        family_metric_frames.append(family_metrics)

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.concat(class_frames, ignore_index=True)
    family_metrics_df = pd.concat(family_metric_frames, ignore_index=True)
    save_df(summary_df, output_dir / "fused_aligned_modality_ablation_summary.csv",
            output_dir / "fused_aligned_modality_ablation_summary.xlsx")
    save_df(class_df, output_dir / "fused_aligned_modality_ablation_per_family.csv",
            output_dir / "fused_aligned_modality_ablation_per_family.xlsx")
    save_df(family_metrics_df, output_dir / "fused_aligned_modality_ablation_per_family_metrics.csv",
            output_dir / "fused_aligned_modality_ablation_per_family_metrics.xlsx")

    mean_pivot = class_df.pivot_table(index="Class", columns="Combination", values="Class Accuracy Mean", aggfunc="first")
    low_pivot = class_df.pivot_table(index="Class", columns="Combination", values="Class Accuracy CI95 Low", aggfunc="first")
    high_pivot = class_df.pivot_table(index="Class", columns="Combination", values="Class Accuracy CI95 High", aggfunc="first")
    single_names = ["Disk", "Memory", "Network"]
    all_name = "Disk+Memory+Network"
    family_rows = []
    for class_idx, family in enumerate(class_names):
        if family not in mean_pivot.index:
            continue
        singles = mean_pivot.loc[family, single_names].dropna()
        best_name = str(singles.idxmax())
        best = float(singles.max())
        single_fold_values = combination_results[best_name]["per_class_fold_metrics"][selected_model_name]["recall"][class_idx]
        for combination, _ in specifications:
            combination_score = float(mean_pivot.loc[family, combination])
            gain = combination_score - best
            gain_low = float(low_pivot.loc[family, combination] - high_pivot.loc[family, best_name])
            gain_high = float(high_pivot.loc[family, combination] - low_pivot.loc[family, best_name])
            combination_fold_values = combination_results[combination]["per_class_fold_metrics"][selected_model_name]["recall"][class_idx]
            try:
                differences = np.asarray(combination_fold_values, dtype=float) - np.asarray(single_fold_values, dtype=float)
                wilcoxon_p = (
                    float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
                    if len(differences) > 1 and np.any(differences != 0) else 1.0
                )
            except Exception:
                wilcoxon_p = np.nan
            permutation_p = paired_sign_permutation_pvalue(combination_fold_values, single_fold_values)
            if combination == best_name:
                guidance = "Reference: strongest aligned single evidence stream for this family."
            elif gain_low > 0:
                guidance = "This combination consistently improved the family relative to its strongest single stream."
            elif gain_high < 0:
                guidance = "This combination consistently reduced the family relative to its strongest single stream."
            else:
                guidance = "The estimated change is uncertain; do not claim a reliable benefit or harm for this family."
            family_rows.append({
                "Family": family,
                "Source combination": combination,
                "Best aligned single source": best_name,
                "Best aligned single-source accuracy": best,
                "Combination accuracy": combination_score,
                "Absolute gain/loss vs strongest single": gain,
                "Percentage gain/loss vs strongest single (%)": float(100.0 * gain / best) if best else np.nan,
                "Conservative difference CI95 Low": gain_low,
                "Conservative difference CI95 High": gain_high,
                "Paired folds": int(min(len(combination_fold_values), len(single_fold_values))),
                "Wilcoxon p-value": wilcoxon_p,
                "Paired permutation p-value": permutation_p,
                "Technical reason status": "Not inferred from performance alone; interpret using class-specific feature/modality importance and raw behavioral reports",
                "Evidence-based guidance": guidance,
            })
    family_df = pd.DataFrame(family_rows).sort_values(
        ["Family", "Absolute gain/loss vs strongest single"], ascending=[True, False]
    ).reset_index(drop=True)
    if not family_df.empty:
        family_df["Wilcoxon Holm-adjusted p-value"] = holm_adjust_pvalues(family_df["Wilcoxon p-value"])
        family_df["Permutation Holm-adjusted p-value"] = holm_adjust_pvalues(family_df["Paired permutation p-value"])
        family_df["Statistically supported at 0.05"] = (
            (family_df["Wilcoxon Holm-adjusted p-value"] < 0.05)
            & (family_df["Permutation Holm-adjusted p-value"] < 0.05)
        )
    save_df(family_df, output_dir / "fused_family_conditional_effects.csv",
            output_dir / "fused_family_conditional_effects.xlsx")
    all_source_df = family_df[family_df["Source combination"] == all_name].copy()
    plot_fusion_gain_bar(
        all_source_df.rename(columns={"Family": "Class", "Absolute gain/loss vs strongest single": "Fusion gain"}),
        output_dir / "fused_family_conditional_effects.png",
        "Exact-ID aligned fusion gain over the best aligned single evidence stream",
    )
    return summary_df, class_df, family_metrics_df, family_df

def save_fused_reproducibility_report(args, bundles: dict, fused_df: pd.DataFrame, X_fused: pd.DataFrame,
                                      fused_models: dict, fusion_info: str, output_dir: Path,
                                      run_started_utc: str, run_started_monotonic: float):
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "shap", "lime"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    timestamp_like = [c for c in X_fused.columns if any(token in c.lower() for token in ["time", "date", "timestamp"])]
    sandbox_metadata = None
    if getattr(args, "sandbox_metadata_json", None):
        with open(args.sandbox_metadata_json, "r", encoding="utf-8") as handle:
            sandbox_metadata = json.load(handle)
        if not isinstance(sandbox_metadata, dict):
            raise ValueError("--sandbox-metadata-json must contain a JSON object.")
    preprocessing_audits = {key: bundles[key].get("preprocessing_audit", {}) for key in FUSION_MODALITIES}
    (output_dir / "fused_source_preprocessing_audit.json").write_text(
        json.dumps(preprocessing_audits, indent=2, default=str), encoding="utf-8"
    )
    report = {
        "scope": "Fused dataset only",
        "fusion_method": fusion_info,
        "random_state": RANDOM_STATE,
        "command_line_arguments": vars(args),
        "python": sys.version,
        "analysis_environment": {
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "network_conditions": "Not measured by this script; do not infer malware-sandbox network conditions from the analysis runtime.",
        },
        "analysis_execution": {
            "started_utc": run_started_utc,
            "report_written_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds_at_report_write": float(time.perf_counter() - run_started_monotonic),
        },
        "packages": packages,
        "malware_sandbox_metadata_supplied_by_user": sandbox_metadata,
        "source_files": {key: bundles[key].get("source_path") for key in FUSION_MODALITIES},
        "source_id_columns": {key: bundles[key].get("id_col") for key in FUSION_MODALITIES},
        "fused_samples": int(len(fused_df)),
        "fused_features": int(X_fused.shape[1]),
        "maximum_selected_features_per_stream": FEATURES_PER_STREAM,
        "source_preprocessing_audits": preprocessing_audits,
        "model_hyperparameters": {name: model.get_params(deep=True) for name, model in fused_models.items()},
        "leakage_controls": [
            "Exact sample identifiers define fused rows.",
            "Grouping uses fusion_id during cross-validation.",
            "Numeric imputation and categorical encoding are fitted only on each training fold.",
            "Mutual-information feature selection is fitted independently within each training fold.",
            "Model-specific scaling and SMOTE are fitted only on each training fold.",
            "Soft-voting model selection and weights use inner CV within each outer training fold.",
        ],
        "unresolved_reviewer_items_requiring_external_evidence": {
            "temporal_concept_drift": (
                "Chronological evaluation is completed only when --fused-time-column names a verified collection/execution timestamp. "
                f"Timestamp-like fused fields detected but not assumed valid: {timestamp_like}."
            ),
            "sandbox_environment_bias": (
                "Malware sandbox conditions are documented only when --sandbox-metadata-json is supplied; the current analysis runtime "
                "must not be presented as the malware execution environment."
            ),
            "feature_scientific_justification": "The script exports provenance and accepts a rationale JSON, but scientific justification must come from the study protocol and literature.",
            "novelty_and_forensic_workflow": "These are manuscript-level claims and cannot be generated from classifier code alone.",
        },
    }
    (output_dir / "fused_reproducibility_and_limitations.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report

def save_reviewer_response_matrix(output_dir: Path, fusion_info: str, args):
    rows = [
        (1, "Exact sample-level alignment", (
             "Implemented for the primary fused benchmark" if fusion_info == "id_exact"
             else "Unavailable in supplied data; pseudo-fusion isolated as sensitivity analysis"
         ),
         "normalize_sample_identifier; build_fused_by_id", "Dataset construction and integrity validation",
         "fused_alignment_audit.csv when exact IDs exist; otherwise fusion_identifier_limitation.json"),
        (2, "Feature extraction and eight-feature justification", "Partially implemented; scientific rationale remains manuscript evidence",
         "load_reduced_dataset_with_id; fit_fold_preprocessor; select_features_by_stream_training_only; save_fused_feature_manifest",
         "Feature extraction, preprocessing, and feature-selection rationale",
         "fused_source_preprocessing_audit.json; fused_feature_manifest.csv; fused_training_fold_feature_selection.csv"),
        (3, "Novelty and related work", "Manuscript action required; code does not invent literature claims",
         "save_reviewer_response_matrix", "Related work and contribution statement",
         "related_work_comparison_template.csv"),
        (4, "Forensic contribution", "Manuscript interpretation required",
         "explainability and ablation outputs", "Forensic workflow, use cases, trust conditions, and limitations",
         "fused_forensic_interpretation_requirements.json"),
        (5, "Explainability and ablation", (
             "Implemented" if fusion_info == "id_exact"
             else "Explainability implemented for sensitivity data; exact-cohort ablation unavailable"
         ),
         "run_explainability_study; run_fused_modality_ablation",
         "Explainability and evidence-ablation results",
         "explainability/; fused_aligned_modality_ablation_summary.csv; fused_representative_correct_incorrect_explanations.csv"),
        (6, "Imbalance and limited scale", "Implemented analytically; generalization discussion remains manuscript work",
         "save_fused_class_balance; run_fused_imbalance_comparison",
         "Class distribution, imbalance treatment, and external-validity limitations",
         "fused_class_balance.csv; fused_imbalance_strategy_comparison.csv; fused_imbalance_strategy_per_family.csv"),
        (7, "Family-conditional fusion effects", (
             "Implemented for exact-ID cohorts; causal/behavioral reasons are not inferred from accuracy alone"
             if fusion_info == "id_exact"
             else "Not executed because a genuine exact-ID-aligned cohort is unavailable"
         ),
         "run_fused_modality_ablation", "Family-conditional results and evidence-based guidance",
         "fused_family_conditional_effects.csv"),
        (8, "Temporal robustness", "Conditional on a verified timestamp",
         "run_fused_temporal_evaluation", "Temporal drift, monitoring, and retraining",
         "fused_temporal_evaluation_status.json and optional temporal metrics"),
        (9, "Sandbox environment bias", "Conditional on externally supplied execution metadata",
         "save_fused_reproducibility_report", "Collection environment and threats to validity",
         "fused_reproducibility_and_limitations.json"),
        (10, "Soft-voting justification", "Implemented with leakage-safe inner-CV weights and descriptive alpha sensitivity",
         "derive_training_only_ensemble_weights; run_fused_ensemble_diagnostics",
         "Ensemble design and sensitivity analysis",
         "fused_ensemble_alpha_sensitivity.csv; fused_model_error_correlation.csv; fused_model_diversity.csv"),
        (11, "Reproducibility", "Implemented for available code/data metadata",
         "save_fused_reproducibility_report", "Reproducibility statement",
         "fused_reproducibility_and_limitations.json; validation_protocol.json"),
    ]
    matrix = pd.DataFrame(rows, columns=[
        "Reviewer item", "Concern", "Implementation status", "Affected code section",
        "Required manuscript section", "Verification evidence",
    ])
    save_df(matrix, output_dir / "reviewer_response_matrix.csv", output_dir / "reviewer_response_matrix.xlsx")

    related_work = pd.DataFrame([
        {
            "Study": "Current exact-ID-aligned fused study",
            "Data alignment": "Exact normalized sample identifier across disk, memory, and network",
            "Modalities": "Disk; memory; network",
            "Fusion level": "Sample-level feature fusion",
            "Classifiers": "Exported from model_hyperparameters in reproducibility report",
            "Explainability": "SHAP; LIME; class-conditioned SRC",
            "Family-level evaluation": "Yes",
            "Forensic contribution": "Must be stated conservatively from generated evidence",
        },
        {
            "Study": "Dambra et al.",
            "Data alignment": "REQUIRES VERIFIED SOURCE",
            "Modalities": "REQUIRES VERIFIED SOURCE",
            "Fusion level": "REQUIRES VERIFIED SOURCE",
            "Classifiers": "REQUIRES VERIFIED SOURCE",
            "Explainability": "REQUIRES VERIFIED SOURCE",
            "Family-level evaluation": "REQUIRES VERIFIED SOURCE",
            "Forensic contribution": "REQUIRES VERIFIED SOURCE",
        },
    ])
    save_df(related_work, output_dir / "related_work_comparison_template.csv",
            output_dir / "related_work_comparison_template.xlsx")

    unresolved = {
        "not_resolved_without_external_material": [
            "Verified details and citations for Dambra et al. and other related work.",
            "Scientific/forensic justification for each supplied feature unless --fused-feature-rationale-json is provided.",
            "Behavioral causes of family-specific gains/losses without raw reports or validated analyst interpretation.",
            "Temporal drift unless --fused-time-column identifies a verified timestamp.",
            "Malware execution environment and sandbox-evasion assessment unless --sandbox-metadata-json is provided.",
            "External generalization beyond the malware families represented in the aligned cohort.",
        ],
        "pseudo_fusion_status": (
            "Not part of the primary benchmark" if fusion_info == "id_exact"
            else "Executed only as explicitly acknowledged sensitivity analysis; must not be called sample-level fusion"
        ),
    }
    (output_dir / "unresolved_reviewer_items.json").write_text(json.dumps(unresolved, indent=2), encoding="utf-8")
    forensic_requirements = {
        "disk": "Interpret only using the actual disk features and raw artefact definitions in the feature manifest/collection protocol.",
        "memory": "Interpret only using the actual memory features and raw artefact definitions in the feature manifest/collection protocol.",
        "network": "Interpret only using the actual network features and raw artefact definitions in the feature manifest/collection protocol.",
        "trust_fusion_when": "Use family-level gains, uncertainty, statistical results, and consistent evidence-stream explanations together.",
        "do_not_trust_fusion_when": "Do not claim benefit where confidence bounds include no gain, corrected tests are not significant, identifiers are not exact, or execution metadata is missing for an environment-sensitive claim.",
    }
    (output_dir / "fused_forensic_interpretation_requirements.json").write_text(
        json.dumps(forensic_requirements, indent=2), encoding="utf-8"
    )
    return matrix

def save_execution_manifest(output_dir: Path, args, fusion_info: str):
    dataset_prefix = "fused" if fusion_info == "id_exact" else "fused pseudo sensitivity"
    explainability_prefix = "fused" if fusion_info == "id_exact" else "fused_pseudo_sensitivity"
    checks = [
        ("Exact alignment audit", "fused_alignment_audit.csv", fusion_info == "id_exact"),
        (("Exact fused model comparison" if fusion_info == "id_exact" else "Pseudo-fusion sensitivity model comparison"),
         f"{dataset_prefix}_all_model_results.csv", True),
        ("Training-fold feature selection", "fused_training_fold_feature_selection.csv", True),
        ("Class-imbalance comparison", "fused_imbalance_strategy_comparison.csv", not args.skip_fused_imbalance_comparison),
        ("Aligned modality ablation", "fused_aligned_modality_ablation_summary.csv", fusion_info == "id_exact" and not args.skip_fused_ablation),
        ("Family-conditional effects", "fused_family_conditional_effects.csv", fusion_info == "id_exact" and not args.skip_fused_ablation),
        ("Ensemble alpha sensitivity", "fused_ensemble_alpha_sensitivity.csv", True),
        ("Model error correlation", "fused_model_error_correlation.csv", True),
        ("Explainability stability", f"explainability/{explainability_prefix}_explainability_summary.csv", True),
        ("Temporal robustness", "fused_temporal_evaluation_metrics.csv", bool(args.fused_time_column)),
    ]
    rows = []
    for experiment, relative_path, requested in checks:
        exists = (output_dir / relative_path).exists()
        rows.append({
            "Experiment": experiment,
            "Requested in this run": bool(requested),
            "Completed": bool(exists),
            "Evidence file": relative_path,
            "Status": "completed" if exists else ("not requested" if not requested else "not completed; inspect logs/status"),
        })
    manifest = pd.DataFrame(rows)
    save_df(manifest, output_dir / "experiment_execution_manifest.csv",
            output_dir / "experiment_execution_manifest.xlsx")
    return manifest

def parse_args():
    parser = argparse.ArgumentParser(description="Reviewer-updated standalone fused malware experiment.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing reduced dataset Excel files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Results folder. Default: results_fused_reviewer_updated.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for XGBoost/CatBoost when available.")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of outer CV folds.")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="Minimum per-class samples to keep.")
    parser.add_argument("--fusion-mode", choices=["auto", "id", "target"], default="auto",
                        help=("Default 'auto' uses exact ID fusion when possible; otherwise it runs a clearly labelled, separately stored "
                              "pseudo-fusion sensitivity analysis and writes an identifier-limitation report. 'id' is strict exact-only. "
                              "'target' explicitly requests pseudo-fusion and requires --allow-pseudo-fusion."))
    parser.add_argument("--allow-pseudo-fusion", action="store_true",
                        help="Required safety acknowledgement for --fusion-mode target. Pseudo-fused output must not be reported as sample-level multimodal fusion.")
    parser.add_argument("--disk-id-column", type=str, default=None)
    parser.add_argument("--memory-id-column", type=str, default=None)
    parser.add_argument("--network-id-column", type=str, default=None)
    parser.add_argument("--fused-time-column", type=str, default=None,
                        help="Verified collection/execution timestamp column for chronological evaluation; it is excluded from predictors.")
    parser.add_argument("--temporal-train-fraction", type=float, default=0.7,
                        help="Earliest fraction used for chronological training when --fused-time-column is provided.")
    parser.add_argument("--sandbox-metadata-json", type=str, default=None,
                        help="Optional verified malware-execution environment metadata (OS, architecture, software, network, duration, settings).")
    parser.add_argument("--fused-feature-rationale-json", type=str, default=None,
                        help="Optional JSON mapping fused/source feature names to scientific rationales for the fused feature manifest.")
    parser.add_argument("--fused-ensemble-alpha", type=str, default="auto",
                        help="Use 'auto' for leakage-safe inner-CV selection from 0, 0.5, 1, 2, and 3, or provide a fixed numeric alpha.")
    parser.add_argument("--skip-fused-ablation", action="store_true",
                        help="Skip the fused aligned single/pair/all modality ablation when only a quick run is needed.")
    parser.add_argument("--skip-fused-imbalance-comparison", action="store_true",
                        help="Skip fused no-resampling versus training-fold-SMOTE comparison.")
    parser.add_argument("--explain-samples", type=int, default=100, help="Number of fused test samples to explain.")
    parser.add_argument("--explain-perturbations", type=int, default=20, help="Number of Gaussian perturbations per sample.")
    parser.add_argument("--explain-perturb-sigma", type=float, default=0.02, help="Gaussian noise scale as a fraction of feature std.")
    parser.add_argument("--src-atoms-per-class", type=int, default=50, help="Maximum SRC dictionary atoms per class.")
    parser.add_argument("--src-nonzero-coefs", type=int, default=10, help="OMP sparsity for SRC reconstruction.")
    parser.add_argument("--explain-test-size", type=float, default=0.2, help="Holdout fraction for explainability evaluation.")
    return parser.parse_args()

def main():
    run_started_monotonic = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    result_root = ensure_dir(
        Path(args.output_dir) if args.output_dir else script_dir / "results_fused_reviewer_updated"
    )
    construction_out = ensure_dir(result_root / "construction")

    search_dirs = build_search_dirs(args.data_dir)
    print_header("SEARCH DIRECTORIES")
    for d in search_dirs:
        print(d)

    bundles = {}
    id_overrides = {
        "disk": args.disk_id_column,
        "memory": args.memory_id_column,
        "network": args.network_id_column,
    }

    for key, filename in DATASET_FILES.items():
        path = find_existing_file(search_dirs, filename)
        bundle = load_reduced_dataset_with_id(path, min_samples=args.min_samples, explicit_id_col=id_overrides[key])
        bundles[key] = bundle
        save_df(bundle["candidate_id_columns"], construction_out / f"{key}_candidate_id_columns.csv",
                construction_out / f"{key}_candidate_id_columns.xlsx")
        print(f"{key}: detected id column -> {bundle['id_col']}")

    # Construct the only evaluated dataset: the fused benchmark.
    fused_df, fusion_info, fusion_audit_df, fusion_conflicts_df, fusion_limitation = (
        construct_fused_dataset(
            bundles,
            fusion_mode=args.fusion_mode,
            allow_pseudo_fusion=bool(args.allow_pseudo_fusion),
        )
    )

    fused_out = result_root if fusion_info == "id_exact" else ensure_dir(result_root / "fused_pseudo_sensitivity")
    save_df(fused_df, fused_out / "fused_dataset.csv", fused_out / "fused_dataset.xlsx")
    if not fusion_audit_df.empty:
        save_df(fusion_audit_df, fused_out / "fused_alignment_audit.csv", fused_out / "fused_alignment_audit.xlsx")
    if not fusion_conflicts_df.empty:
        save_df(fusion_conflicts_df, fused_out / "fused_excluded_label_conflicts.csv",
                fused_out / "fused_excluded_label_conflicts.xlsx")
    if fusion_limitation is not None:
        (fused_out / "fusion_identifier_limitation.json").write_text(
            json.dumps(fusion_limitation, indent=2), encoding="utf-8"
        )
    Path(fused_out / "fusion_method.txt").write_text(
        f"Fusion method used: {fusion_info}\n"
        "id_exact = true sample-id fusion\n"
        "target_pseudo_pairing = class-based pseudo-fusion using target labels only; sensitivity analysis only\n"
        f"Requested mode: {args.fusion_mode}\n"
        "When auto mode cannot establish exact alignment, pseudo-fusion is stored only under "
        "fused_pseudo_sensitivity and is never labelled as genuine sample-level fusion.\n",
        encoding="utf-8",
    )

    X_fused, y_fused, y_raw_fused, class_names_fused = prepare_fused_xy(fused_df)
    temporal_values = None
    if args.fused_time_column:
        if args.fused_time_column not in fused_df.columns:
            raise ValueError(
                f"--fused-time-column '{args.fused_time_column}' was not found. Available columns: {list(fused_df.columns)}"
            )
        temporal_values = fused_df[args.fused_time_column].copy()
        if args.fused_time_column in X_fused.columns:
            X_fused = X_fused.drop(columns=[args.fused_time_column])
    save_fused_feature_manifest(
        X_fused, fused_df, fused_out, rationale_json=args.fused_feature_rationale_json
    )
    save_fused_class_balance(y_raw_fused, fused_out)
    plot_class_counts(y_raw_fused, ensure_dir(fused_out / "plots") / "class_counts_fused.png", f"Fused Class Counts ({fusion_info})")
    fused_models = get_models_for_dataset("fused", use_gpu=args.use_gpu)
    ensemble_alpha = None if str(args.fused_ensemble_alpha).strip().lower() == "auto" else float(args.fused_ensemble_alpha)
    fused_groups = pd.Series(fused_df["fusion_id"].astype(str)) if "fusion_id" in fused_df.columns else None
    fused_dataset_label = "Fused" if fusion_info == "id_exact" else "Fused pseudo sensitivity"
    fused_output = evaluate_dataset(
        dataset_name=fused_dataset_label,
        X=X_fused,
        y=y_fused,
        class_names=class_names_fused,
        output_dir=fused_out,
        models=fused_models,
        cv_folds=args.cv_folds,
        groups=fused_groups,
        ensemble_weight_power=ensemble_alpha,
    )

    run_fused_ensemble_diagnostics(fused_output, class_names_fused, fused_out)

    selected_fused_model_name = str(fused_output["results_df"].iloc[0]["Model"])
    selected_fused_model = fused_models[selected_fused_model_name]
    if not args.skip_fused_imbalance_comparison:
        run_fused_imbalance_comparison(
            X_fused, y_fused, class_names_fused, fused_groups,
            selected_fused_model_name, selected_fused_model, fused_out, args.cv_folds,
        )
    if fusion_info == "id_exact" and not args.skip_fused_ablation:
        run_fused_modality_ablation(
            X_fused, y_fused, class_names_fused, fused_groups,
            selected_fused_model_name, selected_fused_model, fused_output, fused_out,
            args.cv_folds, ensemble_alpha,
        )

    if temporal_values is not None:
        run_fused_temporal_evaluation(
            X_fused, y_fused, class_names_fused, temporal_values,
            selected_fused_model_name, selected_fused_model, fused_out,
            train_fraction=float(args.temporal_train_fraction),
        )
    else:
        (fused_out / "fused_temporal_evaluation_status.json").write_text(
            json.dumps({
                "requested": False,
                "completed": False,
                "reason": "No verified timestamp supplied through --fused-time-column; temporal results were not fabricated.",
            }, indent=2), encoding="utf-8"
        )

    try:
        fused_explainability = run_explainability_study(
            dataset_name=fused_dataset_label,
            X=X_fused,
            y=y_fused,
            class_names=class_names_fused,
            models=fused_models,
            results_df=fused_output["results_df"],
            output_dir=fused_out,
            test_size=float(args.explain_test_size),
            max_samples=int(args.explain_samples),
            perturbations=int(args.explain_perturbations),
            perturb_sigma=float(args.explain_perturb_sigma),
            src_atoms_per_class=int(args.src_atoms_per_class),
            src_nonzero_coefs=int(args.src_nonzero_coefs),
        )
        build_explainability_conclusion({"fused": fused_explainability}, fused_out)
    except Exception as exc:
        print(f"[WARN] Explainability study failed for fused: {exc}")

    save_fused_reproducibility_report(
        args, bundles, fused_df, X_fused, fused_models, fusion_info, fused_out,
        run_started_utc, run_started_monotonic,
    )
    save_reviewer_response_matrix(fused_out, fusion_info, args)
    Path(fused_out / "validation_summary.json").write_text(
        json.dumps({"fused": fused_output.get("validation_info", {})}, indent=2), encoding="utf-8"
    )
    save_execution_manifest(fused_out, args, fusion_info)

    print_header("DONE")
    print(f"Fused output directory: {fused_out}")
    print(f"Fused dataset mode: {fusion_info}")

if __name__ == "__main__":
    main()
