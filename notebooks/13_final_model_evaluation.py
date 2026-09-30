# ============================================================
# 13_final_model_evaluation.py
# Final Model Selection, Unbiased Evaluation, and Pipeline Persistence
# - 76 Final Predictors ("Often feels stressed" excluded)
# - DASS_Stress: Regularized Logistic Regression (C=0.1, class_weight='balanced')
# - DASS-Anxiety: Regularized Logistic Regression (C=0.1, class_weight='balanced')
# - DASS-Depression: Random Forest (n_estimators=300, max_depth=8, min_samples_split=5, class_weight='balanced')
# - 5-Fold Stratified CV (shuffle=True, random_state=42)
# - Fits and persists final full-dataset models in models/
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    recall_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS AND DIRECTORIES
# ============================================================

INPUT_FILE = "data/processed/mental_health_cleaned.xlsx"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

TARGETS = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression"
]

TARGET_MAPPING = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Extremely": 4
}

CLASS_LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe", "Extremely"]


# ============================================================
# 3. FEATURE GROUPS (76 Predictors - "Often feels stressed" excluded)
# ============================================================

binary_yesno_cols = [
    "Taking counseling",
    "Mental illness history in the family",
    "Death in the family",
    "Frequent arguments among parents",
    "Divorced family",
    "Too strict guardians",
    "Frequent conflicts with family members",
    "Addicted family member",
    "Traumatic childhood experience",
    "Quality relationship with family",
    "Emotionally supportive family",
    "Satisfied relationship with family",
    "Biased parents",
    "Family pressure on academic selection",
    "Frequent failure in exam",
    "Worried about academic failure",
    "Family pressure on performing well in exam",
    "Parental satisfaction with academic result",
    "Self-satisfaction with academic performances",
    "Pressure of academic competition",
    "Breakup of a romantic relationship",
    "Insecurity of physical appearance",
    "Fear of communicating with new people",
    "Lack of money to meet basic needs",
    "Difficulty in maintaining a standard lifestyle",
    "Prefer to be alone mostly",
    "Social media monitoring",
    "Social media post review",
    "Unhappy feelings from social media",
    "Lack of personal space at home/ hostel",
    "Unsafe Residence",
    "Dissatisfied Neighborhood",
    "Experience social restrictions",
    "Dissatisfaction with living environment",
    "Physical abuse",
    "Verbal violence/ abuse",
    "Emotional violence/ abuse",
    "Sexual abuse",
    "Social violence/ abuse",
    "Pressure to meet family expectations",
    "Not achieving target/ expectation.",
    "Fear of not achieving life goals/ expectations"
]

binary_other_cols = [
    "Family Type",
    "Educational Institute Type",
    "Personality type"
]

ordinal_3level_cols = [
    "Participation in extracurricular activities",
    "Play video-game",
    "Participate in any sports.",
    "Smoking cigarette",
    "Smoking of marijuana/weed/ alcohol",
    "Taking drugs or any other substance?",
    "Falling sick frequently",
    "Maintaining a balanced diet",
    "Often sleep late at night.",
    "Physical exercise/ Yoga",
    "Religious practices",
    "Often arguments/fights with friends",
    "Bullied/ verbally abused by friends/ others",
    "Cheated by friend",
    "Abused/cheated by partner",
    "Supportive friends",
    "Biased Teacher",
    "Insulted/ harassed by teacher",
    "Inferiority in friendship",
    "Deprived of deservation",
    "Inferiority complex",
    "Time spent on social media",
    "Job Market Insecurity"
]

ordinal_other_cols = [
    "Socio economic status",
    "Academic Year"
]

nominal_cols = [
    "Gender",
    "Residences Area",
    "Marital status",
    "Various struggle to continue study",
    "Educational qualification"
]

numerical_cols = [
    "Age"
]

predictor_groups = (
    binary_yesno_cols
    + binary_other_cols
    + ordinal_3level_cols
    + ordinal_other_cols
    + nominal_cols
    + numerical_cols
)

assert len(predictor_groups) == 76
assert "Often feels stressed" not in predictor_groups


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 13: FINAL MODEL SELECTION & UNBIASED EVALUATION")
print("=" * 75)

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()

categorical_columns = (
    binary_yesno_cols
    + binary_other_cols
    + ordinal_3level_cols
    + ordinal_other_cols
    + nominal_cols
)

for col in categorical_columns:
    df[col] = (
        df[col]
        .astype("object")
        .apply(
            lambda v: v.strip() if isinstance(v, str)
            else np.nan if pd.isna(v)
            else v
        )
    )


# ============================================================
# 5. MEASUREMENT-AWARE PREPROCESSOR
# ============================================================

binary_yesno_categories = [["No", "Yes"] for _ in binary_yesno_cols]
binary_other_categories = [
    ["Single", "Joint"],
    ["Public", "Private"],
    ["Introvert", "Extrovert"]
]
ordinal_3level_categories = [["Never", "Sometimes", "Always"] for _ in ordinal_3level_cols]
ordinal_other_categories = [
    ["Lower", "Lower Middle", "Middle", "Upper Middle", "Upper"],
    ["1st year", "2nd year", "3rd year", "4th year"]
]
nominal_categories = [
    ["Female", "Male", "Other"],
    ["City", "Town", "Village"],
    ["Divorsed", "Married", "Unmarried"],
    ["Family", "Financial", "Mental", "No struggle", "Physical"],
    ["Bachelor", "Diploma", "HSC", "MBBS", "Masters", "Others"]
]

binary_yesno_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=binary_yesno_categories, handle_unknown="use_encoded_value", unknown_value=-1))
])

binary_other_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=binary_other_categories, handle_unknown="use_encoded_value", unknown_value=-1))
])

ordinal_3level_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_3level_categories, handle_unknown="use_encoded_value", unknown_value=-1))
])

ordinal_other_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_other_categories, handle_unknown="use_encoded_value", unknown_value=-1))
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(categories=nominal_categories, handle_unknown="ignore", sparse_output=False))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("binary_yesno", binary_yesno_pipeline, binary_yesno_cols),
        ("binary_other", binary_other_pipeline, binary_other_cols),
        ("ordinal_3level", ordinal_3level_pipeline, ordinal_3level_cols),
        ("ordinal_other", ordinal_other_pipeline, ordinal_other_cols),
        ("nominal", nominal_pipeline, nominal_cols),
        ("numerical", numerical_pipeline, numerical_cols)
    ],
    remainder="drop"
)


# ============================================================
# 6. SELECTED FINAL MODELS PER TARGET
# ============================================================

selected_models = {
    "DASS_Stress": {
        "name": "Logistic Regression (Regularized)",
        "model": LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    },
    "DASS-Anxiety": {
        "name": "Logistic Regression (Regularized)",
        "model": LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    },
    "DASS-Depression": {
        "name": "Random Forest (Cost-Sensitive)",
        "model": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
}


# ============================================================
# 7. UNBIASED 5-FOLD CROSS-VALIDATION EVALUATION
# ============================================================

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
summary_results = []
recall_results = []
confusion_results = []

X = df[predictor_groups].copy()

for target in TARGETS:
    model_info = selected_models[target]
    model_name = model_info["name"]
    estimator = model_info["model"]

    print("\n" + "=" * 75)
    print(f"TARGET: {target} | FINAL MODEL: {model_name}")
    print("=" * 75)

    y = df[target].map(TARGET_MAPPING).astype(int)

    oof_predictions = np.full(len(X), fill_value=-1, dtype=int)
    fold_macro_f1 = []
    fold_balanced_acc = []
    fold_ordinal_mae = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(estimator))
        ])

        # Fit strictly on train fold
        pipeline.fit(X_train, y_train)

        # Predict validation fold
        y_pred = pipeline.predict(X_valid)
        oof_predictions[valid_idx] = y_pred

        macro_f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_valid, y_pred)
        ordinal_mae = mean_absolute_error(y_valid, y_pred)
        recalls = recall_score(y_valid, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)

        fold_macro_f1.append(macro_f1)
        fold_balanced_acc.append(balanced_acc)
        fold_ordinal_mae.append(ordinal_mae)

        fold_results.append({
            "Target": target,
            "Final_Model": model_name,
            "Fold": fold,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced_acc,
            "Ordinal_MAE": ordinal_mae
        })

        for cls_idx, rec in zip(CLASS_LABELS, recalls):
            recall_results.append({
                "Target": target,
                "Final_Model": model_name,
                "Fold": fold,
                "Class": cls_idx,
                "Class_Name": CLASS_NAMES[cls_idx],
                "Recall": rec
            })

        print(
            f"Fold {fold} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Balanced Acc: {balanced_acc:.4f} | "
            f"Ordinal MAE: {ordinal_mae:.4f}"
        )

    # Check OOF completeness
    assert not np.any(oof_predictions == -1)

    # Summary
    summary_results.append({
        "Target": target,
        "Final_Model": model_name,
        "Macro_F1_Mean": np.mean(fold_macro_f1),
        "Macro_F1_STD": np.std(fold_macro_f1, ddof=1),
        "Balanced_Accuracy_Mean": np.mean(fold_balanced_acc),
        "Balanced_Accuracy_STD": np.std(fold_balanced_acc, ddof=1),
        "Ordinal_MAE_Mean": np.mean(fold_ordinal_mae),
        "Ordinal_MAE_STD": np.std(fold_ordinal_mae, ddof=1)
    })

    # Confusion matrix
    cm = confusion_matrix(y, oof_predictions, labels=CLASS_LABELS)
    for t_cls in CLASS_LABELS:
        for p_cls in CLASS_LABELS:
            confusion_results.append({
                "Target": target,
                "Final_Model": model_name,
                "True_Class": t_cls,
                "Predicted_Class": p_cls,
                "Count": cm[t_cls, p_cls]
            })

    oof_recalls = recall_score(y, oof_predictions, labels=CLASS_LABELS, average=None, zero_division=0)
    print(f"\nFinal Out-of-Fold Recalls ({target}):")
    for cls_idx, rec in zip(CLASS_LABELS, oof_recalls):
        print(f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {rec:.4f}")

    # ========================================================
    # 8. FIT FINAL FULL-DATASET MODEL AND SAVE FOR DEPLOYMENT/SHAP
    # ========================================================
    print(f"\nFitting full-dataset model for {target}...")
    final_full_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(estimator))
    ])
    final_full_pipeline.fit(X, y)

    target_clean_name = target.replace("-", "_")
    model_save_path = os.path.join(MODEL_DIR, f"final_model_{target_clean_name}.joblib")
    joblib.dump(final_full_pipeline, model_save_path)
    print(f"Saved full pipeline to {model_save_path}")


# ============================================================
# 9. SAVE EVALUATION ARTIFACTS
# ============================================================

fold_df = pd.DataFrame(fold_results)
summary_df = pd.DataFrame(summary_results)
recall_df = pd.DataFrame(recall_results)
confusion_df = pd.DataFrame(confusion_results)

fold_path = os.path.join(OUTPUT_DIR, "final_model_fold_results.csv")
summary_path = os.path.join(OUTPUT_DIR, "final_model_selection_summary.csv")
recall_path = os.path.join(OUTPUT_DIR, "final_model_class_recall.csv")
confusion_path = os.path.join(OUTPUT_DIR, "final_model_confusion_matrices.csv")

fold_df.to_csv(fold_path, index=False)
summary_df.to_csv(summary_path, index=False)
recall_df.to_csv(recall_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\n" + "=" * 75)
print("STAGE 13 COMPLETE: FINAL MODEL EVALUATION SUMMARY")
print("=" * 75)
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"  - {fold_path}")
print(f"  - {summary_path}")
print(f"  - {recall_path}")
print(f"  - {confusion_path}")
print(f"  - {MODEL_DIR}/final_model_*.joblib")
