# ============================================================
# 09_model_comparison.py
# Corrected Measurement-Aware Model Comparison
# ============================================================

import os
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
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    recall_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
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


# ============================================================
# 3. FEATURE GROUPS
#    Same architecture as corrected 07 and 08
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
    "Often feels stressed",
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


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 5. STRUCTURE VALIDATION
# ============================================================

print("\nValidating feature structure...")

if len(predictor_groups) != len(set(predictor_groups)):
    raise ValueError(
        "Duplicate predictor names found."
    )

missing_predictors = [
    col for col in predictor_groups
    if col not in df.columns
]

if missing_predictors:
    raise ValueError(
        f"Missing predictor columns:\n{missing_predictors}"
    )

missing_targets = [
    target for target in TARGETS
    if target not in df.columns
]

if missing_targets:
    raise ValueError(
        f"Missing target columns:\n{missing_targets}"
    )

for forbidden_col in [
    "Timestamp",
    "Educational Institute Name"
]:

    if forbidden_col in df.columns:
        raise ValueError(
            f"Excluded identifier still present: "
            f"{forbidden_col}"
        )

print("Feature and target validation: PASSED")


# ============================================================
# 6. PREPARE CATEGORICAL DATA
# ============================================================

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
            lambda value:
                value.strip()
                if isinstance(value, str)
                else np.nan
                if pd.isna(value)
                else value
        )
    )


# ============================================================
# 7. CATEGORY VALIDATION
# ============================================================

print("\nValidating categorical values...")

expected_categories = {}

# Binary Yes/No
for col in binary_yesno_cols:
    expected_categories[col] = [
        "No",
        "Yes"
    ]

# Binary Other
expected_categories["Family Type"] = [
    "Single",
    "Joint"
]

expected_categories["Educational Institute Type"] = [
    "Public",
    "Private"
]

expected_categories["Personality type"] = [
    "Introvert",
    "Extrovert"
]

# 3-level ordinal
for col in ordinal_3level_cols:
    expected_categories[col] = [
        "Never",
        "Sometimes",
        "Always"
    ]

# Socio-economic status
expected_categories["Socio economic status"] = [
    "Lower",
    "Lower Middle",
    "Middle",
    "Upper Middle",
    "Upper"
]

# Academic Year
expected_categories["Academic Year"] = [
    "1st year",
    "2nd year",
    "3rd year",
    "4th year"
]


for col, expected in expected_categories.items():

    observed = set(
        df[col]
        .dropna()
        .unique()
    )

    unexpected = observed - set(expected)

    if unexpected:

        raise ValueError(
            f"\nUnexpected categories in '{col}': "
            f"{sorted(unexpected)}\n"
            f"Expected: {expected}"
        )

print("Category validation: PASSED")


# ============================================================
# 8. BUILD MEASUREMENT-AWARE PREPROCESSOR
# ============================================================

binary_yesno_categories = [
    ["No", "Yes"]
    for _ in binary_yesno_cols
]

binary_other_categories = [
    ["Single", "Joint"],
    ["Public", "Private"],
    ["Introvert", "Extrovert"]
]

ordinal_3level_categories = [
    ["Never", "Sometimes", "Always"]
    for _ in ordinal_3level_cols
]

ordinal_other_categories = [
    [
        "Lower",
        "Lower Middle",
        "Middle",
        "Upper Middle",
        "Upper"
    ],
    [
        "1st year",
        "2nd year",
        "3rd year",
        "4th year"
    ]
]


binary_yesno_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=binary_yesno_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


binary_other_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=binary_other_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


ordinal_3level_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=ordinal_3level_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


ordinal_other_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=ordinal_other_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


nominal_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "binary_yesno",
            binary_yesno_pipeline,
            binary_yesno_cols
        ),
        (
            "binary_other",
            binary_other_pipeline,
            binary_other_cols
        ),
        (
            "ordinal_3level",
            ordinal_3level_pipeline,
            ordinal_3level_cols
        ),
        (
            "ordinal_other",
            ordinal_other_pipeline,
            ordinal_other_cols
        ),
        (
            "nominal",
            nominal_pipeline,
            nominal_cols
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_cols
        )
    ],
    remainder="drop"
)


# ============================================================
# 9. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ============================================================
# 10. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 11. RESULT STORAGE
# ============================================================

fold_results = []
summary_results = []
recall_results = []
confusion_results = []


# ============================================================
# 12. TARGET LOOP
# ============================================================

for target in TARGETS:

    print("\n" + "=" * 70)
    print(f"TARGET: {target}")
    print("=" * 70)

    X = df[predictor_groups].copy()

    y = df[target].map(
        TARGET_MAPPING
    )

    if y.isna().any():

        raise ValueError(
            f"Unknown target labels found "
            f"in {target}."
        )

    y = y.astype(int)

    print("\nTarget distribution:")
    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )


    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Model: {model_name}")
        print("-" * 70)

        oof_pred = np.full(
            len(X),
            -1,
            dtype=int
        )

        fold_macro_f1 = []
        fold_balanced_acc = []
        fold_ordinal_mae = []

        # ====================================================
        # FOLD LOOP
        # ====================================================

        for fold, (
            train_idx,
            valid_idx
        ) in enumerate(
            cv.split(X, y),
            start=1
        ):

            X_train = X.iloc[
                train_idx
            ].copy()

            X_valid = X.iloc[
                valid_idx
            ].copy()

            y_train = y.iloc[
                train_idx
            ]

            y_valid = y.iloc[
                valid_idx
            ]


            # Fresh pipeline every fold
            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(preprocessor)
                    ),
                    (
                        "model",
                        clone(model)
                    )
                ]
            )


            # Fit ONLY on training fold
            pipeline.fit(
                X_train,
                y_train
            )


            # Predict validation fold
            y_pred = pipeline.predict(
                X_valid
            )


            # OOF storage
            oof_pred[
                valid_idx
            ] = y_pred


            # =================================================
            # METRICS
            # =================================================

            macro_f1 = f1_score(
                y_valid,
                y_pred,
                average="macro",
                zero_division=0
            )

            balanced_acc = (
                balanced_accuracy_score(
                    y_valid,
                    y_pred
                )
            )

            ordinal_mae = (
                mean_absolute_error(
                    y_valid,
                    y_pred
                )
            )

            recalls = recall_score(
                y_valid,
                y_pred,
                labels=CLASS_LABELS,
                average=None,
                zero_division=0
            )


            fold_macro_f1.append(
                macro_f1
            )

            fold_balanced_acc.append(
                balanced_acc
            )

            fold_ordinal_mae.append(
                ordinal_mae
            )


            # =================================================
            # STORE FOLD RESULTS
            # =================================================

            fold_results.append(
                {
                    "Target": target,
                    "Model": model_name,
                    "Fold": fold,
                    "Macro_F1": macro_f1,
                    "Balanced_Accuracy": balanced_acc,
                    "Ordinal_MAE": ordinal_mae
                }
            )


            # =================================================
            # STORE CLASS RECALL
            # =================================================

            for class_idx, recall_value in zip(
                CLASS_LABELS,
                recalls
            ):

                recall_results.append(
                    {
                        "Target": target,
                        "Model": model_name,
                        "Fold": fold,
                        "Class": class_idx,
                        "Recall": recall_value
                    }
                )


            print(
                f"Fold {fold}: "
                f"Macro-F1={macro_f1:.4f}, "
                f"Balanced Acc={balanced_acc:.4f}, "
                f"Ordinal MAE={ordinal_mae:.4f}"
            )


        # ====================================================
        # OOF INTEGRITY CHECK
        # ====================================================

        if np.any(oof_pred == -1):

            raise RuntimeError(
                f"Missing OOF predictions for "
                f"{target} - {model_name}."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        summary_results.append(
            {
                "Target": target,
                "Model": model_name,

                "Macro_F1_Mean":
                    np.mean(
                        fold_macro_f1
                    ),

                "Macro_F1_STD":
                    np.std(
                        fold_macro_f1,
                        ddof=1
                    ),

                "Balanced_Accuracy_Mean":
                    np.mean(
                        fold_balanced_acc
                    ),

                "Balanced_Accuracy_STD":
                    np.std(
                        fold_balanced_acc,
                        ddof=1
                    ),

                "Ordinal_MAE_Mean":
                    np.mean(
                        fold_ordinal_mae
                    ),

                "Ordinal_MAE_STD":
                    np.std(
                        fold_ordinal_mae,
                        ddof=1
                    )
            }
        )


        # ====================================================
        # OOF CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            y,
            oof_pred,
            labels=CLASS_LABELS
        )

        for true_class in CLASS_LABELS:

            for predicted_class in CLASS_LABELS:

                confusion_results.append(
                    {
                        "Target": target,
                        "Model": model_name,
                        "True_Class": true_class,
                        "Predicted_Class": predicted_class,
                        "Count": cm[
                            true_class,
                            predicted_class
                        ]
                    }
                )


        # ====================================================
        # OOF CLASS RECALL
        # ====================================================

        oof_recalls = recall_score(
            y,
            oof_pred,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0
        )

        print("\nOOF class recall:")

        for class_idx, recall_value in zip(
            CLASS_LABELS,
            oof_recalls
        ):

            print(
                f"Class {class_idx}: "
                f"{recall_value:.4f}"
            )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

summary_results_df = pd.DataFrame(
    summary_results
)

recall_results_df = pd.DataFrame(
    recall_results
)

confusion_results_df = pd.DataFrame(
    confusion_results
)


fold_output = (
    f"{OUTPUT_DIR}/model_comparison_fold_results_corrected.csv"
)

summary_output = (
    f"{OUTPUT_DIR}/model_comparison_results_corrected.csv"
)

recall_output = (
    f"{OUTPUT_DIR}/model_comparison_class_recall_corrected.csv"
)

confusion_output = (
    f"{OUTPUT_DIR}/model_comparison_confusion_matrices_corrected.csv"
)


fold_results_df.to_csv(
    fold_output,
    index=False
)

summary_results_df.to_csv(
    summary_output,
    index=False
)

recall_results_df.to_csv(
    recall_output,
    index=False
)

confusion_results_df.to_csv(
    confusion_output,
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED MODEL COMPARISON COMPLETE")
print("=" * 70)

print("\nModel comparison summary:")

print(
    summary_results_df.to_string(
        index=False
    )
)

print("\nFiles saved:")
print(f"- {fold_output}")
print(f"- {summary_output}")
print(f"- {recall_output}")
print(f"- {confusion_output}")

print("\nAll model comparison checks PASSED.")