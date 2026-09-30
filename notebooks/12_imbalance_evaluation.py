# ============================================================
# 12_imbalance_evaluation.py
# Imbalance-Strategy Evaluation
# - 76 Final Predictors ("Often feels stressed" excluded)
# - Evaluates: None, Cost-Sensitive (Balanced), Random Over-Sampling, SMOTE
# - Models: Logistic Regression & Random Forest
# - 5-Fold StratifiedKFold (shuffle=True, random_state=42)
# - Resampling applied STRICTLY inside training folds
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

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
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

assert len(predictor_groups) == 76, f"Expected 76 predictors, found {len(predictor_groups)}"
assert "Often feels stressed" not in predictor_groups


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 12: IMBALANCE-STRATEGY EVALUATION")
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
# 6. DEFINE IMBALANCE STRATEGIES & PIPELINE BUILDER
# ============================================================

def build_pipeline(strategy, model_type):
    """
    Constructs an imbalanced-learn pipeline ensuring:
    - Preprocessor fitted only on training fold.
    - Resampling applied only on training fold.
    - Model trained on resampled/reweighted data.
    - Test/Validation fold passed ONLY through preprocessor transform.
    """
    steps = [("preprocessor", clone(preprocessor))]

    # 1. Resampling step (if applicable)
    if strategy == "Random_OverSampler":
        steps.append(("resampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif strategy == "SMOTE":
        # k_neighbors=3 because smallest class in training fold has ~11 samples
        steps.append(("resampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))

    # 2. Classifier step
    if model_type == "Logistic Regression":
        if strategy == "Cost_Sensitive_Balanced":
            clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE, C=0.1)
        else:
            clf = LogisticRegression(class_weight=None, max_iter=3000, random_state=RANDOM_STATE, C=0.1)
    elif model_type == "Random Forest":
        if strategy == "Cost_Sensitive_Balanced":
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=8, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
        elif strategy == "Balanced_Subsample":
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", max_depth=8, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_estimators=300, class_weight=None, max_depth=8, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)

    steps.append(("model", clf))
    return ImbPipeline(steps)


strategies = {
    "Logistic Regression": [
        "None_Unweighted",
        "Cost_Sensitive_Balanced",
        "Random_OverSampler",
        "SMOTE"
    ],
    "Random Forest": [
        "None_Unweighted",
        "Cost_Sensitive_Balanced",
        "Balanced_Subsample",
        "Random_OverSampler",
        "SMOTE"
    ]
}


# ============================================================
# 7. CROSS-VALIDATION LOOP
# ============================================================

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
summary_results = []
recall_results = []
confusion_results = []

for target in TARGETS:
    print("\n" + "=" * 75)
    print(f"EVALUATING IMBALANCE STRATEGIES FOR TARGET: {target}")
    print("=" * 75)

    X = df[predictor_groups].copy()
    y = df[target].map(TARGET_MAPPING).astype(int)

    for model_name, strat_list in strategies.items():
        print(f"\n--- Model Family: {model_name} ---")

        for strategy_name in strat_list:
            print(f"\nEvaluating Strategy: {strategy_name}")

            oof_predictions = np.full(len(X), fill_value=-1, dtype=int)
            fold_macro_f1 = []
            fold_balanced_acc = []
            fold_ordinal_mae = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
                X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                pipe = build_pipeline(strategy_name, model_name)

                # Fit pipeline: Preprocessor -> Resampler -> Model strictly on fold train
                pipe.fit(X_train, y_train)

                # Validation transform & predict (resampler is bypassed during predict!)
                y_pred = pipe.predict(X_valid)
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
                    "Model": model_name,
                    "Strategy": strategy_name,
                    "Fold": fold,
                    "Macro_F1": macro_f1,
                    "Balanced_Accuracy": balanced_acc,
                    "Ordinal_MAE": ordinal_mae
                })

                for cls_idx, rec in zip(CLASS_LABELS, recalls):
                    recall_results.append({
                        "Target": target,
                        "Model": model_name,
                        "Strategy": strategy_name,
                        "Fold": fold,
                        "Class": cls_idx,
                        "Class_Name": CLASS_NAMES[cls_idx],
                        "Recall": rec
                    })

            # Check OOF completeness
            assert not np.any(oof_predictions == -1)

            # Strategy summary
            summary_results.append({
                "Target": target,
                "Model": model_name,
                "Strategy": strategy_name,
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
                        "Model": model_name,
                        "Strategy": strategy_name,
                        "True_Class": t_cls,
                        "Predicted_Class": p_cls,
                        "Count": cm[t_cls, p_cls]
                    })

            oof_recalls = recall_score(y, oof_predictions, labels=CLASS_LABELS, average=None, zero_division=0)
            print(
                f"  Mean Macro-F1: {np.mean(fold_macro_f1):.4f} | "
                f"Bal Acc: {np.mean(fold_balanced_acc):.4f} | "
                f"Ord MAE: {np.mean(fold_ordinal_mae):.4f}"
            )
            print(f"  OOF Recalls: Normal={oof_recalls[0]:.2f}, Mild={oof_recalls[1]:.2f}, Mod={oof_recalls[2]:.2f}, Sev={oof_recalls[3]:.2f}, Ext={oof_recalls[4]:.2f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

fold_df = pd.DataFrame(fold_results)
summary_df = pd.DataFrame(summary_results)
recall_df = pd.DataFrame(recall_results)
confusion_df = pd.DataFrame(confusion_results)

fold_path = os.path.join(OUTPUT_DIR, "imbalance_evaluation_fold_results.csv")
summary_path = os.path.join(OUTPUT_DIR, "imbalance_evaluation_summary.csv")
recall_path = os.path.join(OUTPUT_DIR, "imbalance_evaluation_class_recall.csv")
confusion_path = os.path.join(OUTPUT_DIR, "imbalance_evaluation_confusion_matrices.csv")

fold_df.to_csv(fold_path, index=False)
summary_df.to_csv(summary_path, index=False)
recall_df.to_csv(recall_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\n" + "=" * 75)
print("STAGE 12 COMPLETE: IMBALANCE EVALUATION SUMMARY")
print("=" * 75)
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"  - {fold_path}")
print(f"  - {summary_path}")
print(f"  - {recall_path}")
print(f"  - {confusion_path}")
