import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from pathlib import Path


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_excel("data/processed/mental_health_cleaned.xlsx")

targets = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression"
]


# ============================================================
# 2. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    col
    for col in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    if col not in targets
]

print("Number of categorical features:", len(categorical_features))
print("Features:")
print(categorical_features)


# ============================================================
# 3. CHI-SQUARE + CRAMER'S V
# ============================================================

results = []

for feature in categorical_features:

    for target in targets:

        # Create contingency table
        table = pd.crosstab(
            df[feature],
            df[target]
        )

        # Skip if there are not enough categories
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(
            table,
            correction=False
        )

        # Number of observations
        n = table.to_numpy().sum()

        # Number of rows and columns
        r, c = table.shape

        # Cramer's V
        cramers_v = np.sqrt(
            chi2 / (n * min(r - 1, c - 1))
        )

        # Check expected frequencies
        expected_flat = expected.ravel()

        low_expected_count = int(
            (expected_flat < 5).sum()
        )

        low_expected_pct = (
            low_expected_count /
            len(expected_flat)
        ) * 100

        results.append({
            "feature": feature,
            "target": target,
            "chi2": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "cramers_v": cramers_v,
            "min_expected": expected_flat.min(),
            "low_expected_pct": low_expected_pct
        })


# Convert results to DataFrame
results_df = pd.DataFrame(results)


# ============================================================
# 4. FDR CORRECTION
# ============================================================

results_df["p_value_fdr"] = multipletests(
    results_df["p_value"],
    method="fdr_bh"
)[1]

results_df["significant_fdr_0.05"] = (
    results_df["p_value_fdr"] < 0.05
)


# ============================================================
# 5. FLAG SPARSE TABLES
# ============================================================

results_df["sparse_table_warning"] = (
    results_df["low_expected_pct"] > 20
)


# ============================================================
# 6. SORT RESULTS
# ============================================================

results_df = results_df.sort_values(
    by=["p_value_fdr", "cramers_v"],
    ascending=[True, False]
)


# ============================================================
# 7. SAVE COMPLETE RESULTS
# ============================================================

Path("outputs").mkdir(exist_ok=True)

output_path = (
    "outputs/categorical_association_tests.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 8. DISPLAY SUMMARY
# ============================================================

print("\n==========================================")
print("STATISTICAL ASSOCIATION TESTING")
print("==========================================")

print(
    "\nTotal tests performed:",
    len(results_df)
)

print(
    "\nSignificant associations after FDR correction:",
    results_df["significant_fdr_0.05"].sum()
)


print("\n==========================================")
print("TOP 20 ASSOCIATIONS")
print("==========================================")

display_columns = [
    "feature",
    "target",
    "cramers_v",
    "p_value",
    "p_value_fdr",
    "significant_fdr_0.05",
    "sparse_table_warning"
]

print(
    results_df[display_columns]
    .head(20)
    .to_string(index=False)
)


print("\n==========================================")
print("RESULT FILE")
print("==========================================")

print(
    "Saved to:",
    output_path
)