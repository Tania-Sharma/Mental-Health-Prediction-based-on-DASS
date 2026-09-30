# ============================================================
# 11_nested_tuning.py
# Nested Cross-Validation Hyperparameter Tuning
# - 76 Final Predictors ("Often feels stressed" excluded)
# - Outer 5-fold Stratified CV (Unbiased evaluation)
# - Inner 3-fold Stratified CV (Hyperparameter selection)
# - Random state = 42
# ============================================================

import os
import json
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

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)
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
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS & REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3

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

# 23 variables: "Often feels stressed" excluded
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

assert len(predictor_groups) == 76, f"Expected 76 predictors, got {len(predictor_groups)}"
assert "Often feels stressed" not in predictor_groups, "Often feels stressed must be excluded"


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 75)
print("STAGE 11: NESTED HYPERPARAMETER TUNING & EVALUATION")
print("=" * 75)

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()
print(f"Dataset shape: {df.shape}")
print(f"Total Final Predictors: {len(predictor_groups)}")

# Standardize whitespace in categorical features
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

# Survey-defined categories for nominal variables to guarantee deterministic feature dimensionality
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
# 6. MODEL SEARCH SPACES (TUNING GRIDS)
# ============================================================

param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
        "model__class_weight": ["balanced", None]
    },
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": ["balanced", "balanced_subsample"]
    }
}

base_estimators = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 7. NESTED CROSS-VALIDATION EXECUTION
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
summary_results = []
recall_results = []
confusion_results = []
best_params_records = []

for target in TARGETS:
    print("\n" + "=" * 75)
    print(f"EVALUATING TARGET: {target}")
    print("=" * 75)

    X = df[predictor_groups].copy()
    y = df[target].map(TARGET_MAPPING).astype(int)

    print("\nTarget Class Distribution:")
    for cls_idx, count in y.value_counts().sort_index().items():
        print(f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {count} ({count/len(y)*100:.1f}%)")

    for model_name, base_estimator in base_estimators.items():
        print("\n" + "-" * 75)
        print(f"Model: {model_name} (Nested 5x3 Stratified CV)")
        print("-" * 75)

        oof_predictions = np.full(len(X), fill_value=-1, dtype=int)
        fold_macro_f1 = []
        fold_balanced_acc = []
        fold_ordinal_mae = []

        for outer_fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Full pipeline: Preprocessor + Estimator
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(base_estimator))
            ])

            # Inner CV for hyperparameter tuning (fitted ONLY on X_train)
            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE + outer_fold
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=-1,
                refit=True
            )

            # Fit inner tuning and automatically refit best estimator on full outer training fold
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_inner_score = grid_search.best_score_

            # Clean parameter keys for reporting
            cleaned_params = {k.replace("model__", ""): v for k, v in best_params.items()}

            # Unbiased evaluation on holdout outer validation fold
            y_pred = best_model.predict(X_valid)
            oof_predictions[valid_idx] = y_pred

            # Metrics
            macro_f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(y_valid, y_pred)
            ordinal_mae = mean_absolute_error(y_valid, y_pred)
            recalls = recall_score(y_valid, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)

            fold_macro_f1.append(macro_f1)
            fold_balanced_acc.append(balanced_acc)
            fold_ordinal_mae.append(ordinal_mae)

            print(
                f"Outer Fold {outer_fold} | "
                f"Inner Macro-F1: {best_inner_score:.4f} | "
                f"Outer Macro-F1: {macro_f1:.4f} | "
                f"Bal Acc: {balanced_acc:.4f} | "
                f"Ord MAE: {ordinal_mae:.4f}"
            )
            print(f"  Selected Params: {cleaned_params}")

            fold_results.append({
                "Target": target,
                "Model": model_name,
                "Outer_Fold": outer_fold,
                "Best_Inner_Macro_F1": best_inner_score,
                "Macro_F1": macro_f1,
                "Balanced_Accuracy": balanced_acc,
                "Ordinal_MAE": ordinal_mae,
                "Best_Parameters": json.dumps(cleaned_params)
            })

            best_params_records.append({
                "Target": target,
                "Model": model_name,
                "Outer_Fold": outer_fold,
                **cleaned_params
            })

            for cls_idx, rec in zip(CLASS_LABELS, recalls):
                recall_results.append({
                    "Target": target,
                    "Model": model_name,
                    "Outer_Fold": outer_fold,
                    "Class": cls_idx,
                    "Class_Name": CLASS_NAMES[cls_idx],
                    "Recall": rec
                })

        # Check OOF completeness
        assert not np.any(oof_predictions == -1), f"Missing OOF predictions for {target} - {model_name}"

        # Summary over 5 outer folds
        summary_results.append({
            "Target": target,
            "Model": model_name,
            "Macro_F1_Mean": np.mean(fold_macro_f1),
            "Macro_F1_STD": np.std(fold_macro_f1, ddof=1),
            "Balanced_Accuracy_Mean": np.mean(fold_balanced_acc),
            "Balanced_Accuracy_STD": np.std(fold_balanced_acc, ddof=1),
            "Ordinal_MAE_Mean": np.mean(fold_ordinal_mae),
            "Ordinal_MAE_STD": np.std(fold_ordinal_mae, ddof=1)
        })

        # Out-of-fold confusion matrix
        cm = confusion_matrix(y, oof_predictions, labels=CLASS_LABELS)
        for t_cls in CLASS_LABELS:
            for p_cls in CLASS_LABELS:
                confusion_results.append({
                    "Target": target,
                    "Model": model_name,
                    "True_Class": t_cls,
                    "Predicted_Class": p_cls,
                    "Count": cm[t_cls, p_cls]
                })

        oof_recalls = recall_score(y, oof_predictions, labels=CLASS_LABELS, average=None, zero_division=0)
        print(f"\nOverall OOF Recalls ({target} - {model_name}):")
        for cls_idx, rec in zip(CLASS_LABELS, oof_recalls):
            print(f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {rec:.4f}")


# ============================================================
# 8. SAVE ARTIFACTS
# ============================================================

fold_df = pd.DataFrame(fold_results)
summary_df = pd.DataFrame(summary_results)
recall_df = pd.DataFrame(recall_results)
confusion_df = pd.DataFrame(confusion_results)
params_df = pd.DataFrame(best_params_records)

fold_path = os.path.join(OUTPUT_DIR, "nested_tuning_fold_results.csv")
summary_path = os.path.join(OUTPUT_DIR, "nested_tuning_summary.csv")
recall_path = os.path.join(OUTPUT_DIR, "nested_tuning_class_recall.csv")
confusion_path = os.path.join(OUTPUT_DIR, "nested_tuning_confusion_matrices.csv")
params_path = os.path.join(OUTPUT_DIR, "nested_tuning_selected_parameters.csv")

fold_df.to_csv(fold_path, index=False)
summary_df.to_csv(summary_path, index=False)
recall_df.to_csv(recall_path, index=False)
confusion_df.to_csv(confusion_path, index=False)
params_df.to_csv(params_path, index=False)

print("\n" + "=" * 75)
print("STAGE 11 COMPLETE: NESTED HYPERPARAMETER TUNING SUMMARY")
print("=" * 75)
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"  - {fold_path}")
print(f"  - {summary_path}")
print(f"  - {recall_path}")
print(f"  - {confusion_path}")
print(f"  - {params_path}")
