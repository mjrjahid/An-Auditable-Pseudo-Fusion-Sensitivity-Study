# Malware Family Classification Using Pseudo-Fusion

This experiment studies how disk, memory, and network data can help classify malware into known families. It uses existing VirusTotal and Hybrid Analysis reports, without running malware locally.

## Data

- **Initial collection:** 14,400 MalwareBazaar hashes from 33 families, collected from January to April 2026.
- **Study dataset:** 1,044 constructed records from 18 families.
- **Features:** 21 total, with seven from each data source.

**What pseudo-fusion means:** Records are paired within the same malware family. A combined row may contain data from different malware samples. The results show performance on these constructed rows; they do not prove a benefit from combining evidence from the same sample.

## Experiment

1. Check data columns, sample IDs, duplicates, and family labels.
2. Pair records within each family, using the smallest available count across the three sources.
3. Compare Logistic Regression, Random Forest, RBF-SVM, KNN, XGBoost, and CatBoost.
4. Use 10-fold stratified group cross-validation. Fit preprocessing and SMOTE only on training data.
5. Test 30 alternative pairings, compare results with and without SMOTE, and evaluate a soft-voting ensemble.
6. Review model explanations using SHAP, LIME, and sparse representation.

Pairing IDs and construction-related fields are excluded from model inputs.

## Reported Results

| Random Forest test | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Seed 42 | 88.02% | 0.7889 |
| Average of 30 alternative pairings | 86.63% | 0.7661 |

Random Forest performed best among the tested classifiers. Its seed-42 balanced accuracy was **78.59%**. Pairing choices changed the results, SMOTE gave small mixed benefits, and SHAP gave the most stable explanations.

## Software

Python 3.12.8, NumPy, Pandas, SciPy, scikit-learn, imbalanced-learn, XGBoost, CatBoost, SHAP, and LIME.
