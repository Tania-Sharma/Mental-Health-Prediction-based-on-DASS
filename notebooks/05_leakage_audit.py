import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path


# ==========================================
# 1. LOAD DATA
# ==========================================

input_path = "data/processed/mental_health_cleaned.xlsx"

df = pd.read_excel(input_path)

print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE TARGETS
# ==========================================

targets = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression"
]


# ==========================================
# 3. DEFINE DEFINITE EXCLUSIONS
# ==========================================

known_exclusions = {
    "Timestamp":
        "Collection timestamp; not a meaningful pre-outcome predictor",

    "DASS_Stress":
        "Target variable",

    "DASS-Anxiety":
        "Target variable",

    "DASS-Depression":
        "Target variable"
}


# ==========================================
# 4. DEFINE FEATURES REQUIRING REVIEW
# ==========================================

review_features = {

    "Taking counseling":
        "May reflect existing mental-health concerns; timing of counseling must be verified",

    "Often feels stressed":
        "Strong conceptual overlap with DASS Stress; requires detailed assessment",

    "Inferiority complex":
        "Psychological construct that may overlap with mental-health outcomes",

    "Unhappy feelings from social media":
        "Emotional/psychological construct; possible overlap with mental-health outcomes",

    "Deprived of deservation":
        "Variable meaning and measurement need clarification",

    "Falling sick frequently":
        "Potentially valid health-related predictor; temporal direction should be checked",

    "Personality type":
        "Potentially valid predictor; measurement definition should be verified"
}


# ==========================================
# 5. INITIAL COLUMN-LEVEL AUDIT
# ==========================================

audit_results = []

for column in df.columns:

    if column in targets:

        category = "TARGET"
        reason = known_exclusions[column]

    elif column == "Timestamp":

        category = "EXCLUDE"
        reason = known_exclusions[column]

    elif column in review_features:

        category = "REQUIRES REVIEW"
        reason = review_features[column]

    else:

        category = "POTENTIALLY VALID"

        reason = (
            "No obvious direct target information identified "
            "from the feature name. Requires temporal and "
            "construct-level validation."
        )

    audit_results.append({
        "feature": column,
        "category": category,
        "reason": reason
    })


audit_df = pd.DataFrame(audit_results)


# ==========================================
# 6. DETAILED ANALYSIS:
#    OFTEN FEELS STRESSED
# ==========================================

print("\n==========================================")
print("DETAILED CHECK: OFTEN FEELS STRESSED")
print("==========================================")


stress_map = {
    "Never": 0,
    "Sometimes": 1,
    "Always": 2
}


# Convert DASS Stress severity into ordinal codes
order = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Extremely"
]

df["DASS_Stress"] = pd.Categorical(
    df["DASS_Stress"],
    categories=order,
    ordered=True
)

dass_stress_codes = df["DASS_Stress"].cat.codes

self_report_numeric = (
    df["Often feels stressed"]
    .map(stress_map)
)


# ------------------------------------------
# Check 1: Spearman correlation
# ------------------------------------------

corr, pval = spearmanr(
    self_report_numeric,
    dass_stress_codes
)

print(
    f"\nSpearman correlation: "
    f"{corr:.3f}"
)

print(
    f"p-value: {pval:.5f}"
)

print(
    "\nInterpretation:"
)

print(
    "The correlation measures how strongly self-reported "
    "stress severity changes with DASS Stress severity."
)


# ------------------------------------------
# Check 2: Baseline comparison
# ------------------------------------------

baseline_col = "Play video-game"

baseline_map = {
    "Never": 0,
    "Sometimes": 1,
    "Always": 2
}

baseline_numeric = (
    df[baseline_col]
    .map(baseline_map)
)

baseline_corr, baseline_pval = spearmanr(
    baseline_numeric,
    dass_stress_codes
)

print(
    f"\nBaseline feature: {baseline_col}"
)

print(
    f"Baseline correlation with DASS Stress: "
    f"{baseline_corr:.3f}"
)

print(
    f"'Often feels stressed' correlation with DASS Stress: "
    f"{corr:.3f}"
)


# ------------------------------------------
# Check 3: High-stress agreement
# ------------------------------------------

always_group = df[
    df["Often feels stressed"] == "Always"
]

high_dass = (
    always_group["DASS_Stress"]
    .isin(["Severe", "Extremely"])
    .mean()
)

print(
    f"\nStudents reporting 'Always' stressed: "
    f"{len(always_group)}"
)

print(
    f"Among them, Severe/Extremely DASS Stress: "
    f"{high_dass * 100:.1f}%"
)


# ------------------------------------------
# Check 4: Full cross-tabulation
# ------------------------------------------

crosstab = pd.crosstab(
    df["Often feels stressed"],
    df["DASS_Stress"]
)

print(
    "\nCross-tabulation:"
)

print(crosstab)


# ------------------------------------------
# Check 5: Row percentages
# ------------------------------------------

row_percentages = pd.crosstab(
    df["Often feels stressed"],
    df["DASS_Stress"],
    normalize="index"
).round(3)

print(
    "\nRow percentages:"
)

print(row_percentages)


# ==========================================
# 7. DETAILED REVIEW DECISION
# ==========================================

print("\n==========================================")
print("LEAKAGE INTERPRETATION")
print("==========================================")

print(
    "\n'Often feels stressed' is NOT automatically "
    "classified as confirmed leakage."
)

print(
    "However, it is flagged for review because its "
    "construct is closely related to DASS Stress."
)

print(
    "\nThe final decision should consider:"
)

print(
    "1. Whether the variable was collected before the "
    "DASS assessment."
)

print(
    "2. Whether the variable was used to calculate "
    "the DASS outcome."
)

print(
    "3. Whether the variable essentially asks the same "
    "construct as the DASS Stress items."
)

print(
    "4. Whether the feature would realistically be "
    "available when the model makes a prediction."
)


# ==========================================
# 8. CREATE DETAILED REVIEW TABLE
# ==========================================

review_results = []

for feature, reason in review_features.items():

    review_results.append({
        "feature": feature,
        "review_reason": reason,
        "status": "Requires manual/temporal assessment"
    })


review_df = pd.DataFrame(review_results)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

Path("outputs").mkdir(exist_ok=True)


audit_output = (
    "outputs/leakage_audit.csv"
)

review_output = (
    "outputs/leakage_review_features.csv"
)


audit_df.to_csv(
    audit_output,
    index=False
)


review_df.to_csv(
    review_output,
    index=False
)


# ==========================================
# 10. SUMMARY
# ==========================================

print("\n==========================================")
print("LEAKAGE AUDIT SUMMARY")
print("==========================================")

print(
    "\nCategory counts:"
)

print(
    audit_df["category"]
    .value_counts()
)


print(
    "\nTotal columns audited:",
    len(audit_df)
)

print(
    "Targets:",
    len(
        audit_df[
            audit_df["category"] == "TARGET"
        ]
    )
)

print(
    "Excluded:",
    len(
        audit_df[
            audit_df["category"] == "EXCLUDE"
        ]
    )
)

print(
    "Requires review:",
    len(
        audit_df[
            audit_df["category"] == "REQUIRES REVIEW"
        ]
    )
)

print(
    "Potentially valid:",
    len(
        audit_df[
            audit_df["category"] == "POTENTIALLY VALID"
        ]
    )
)


print("\n==========================================")
print("OUTPUT FILES")
print("==========================================")

print(
    "Full audit:",
    audit_output
)

print(
    "Review features:",
    review_output
)

print("\nLeakage audit completed.")