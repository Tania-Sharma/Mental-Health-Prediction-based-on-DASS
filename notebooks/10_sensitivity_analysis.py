# ============================================================
# 10_sensitivity_analysis.py
# Sensitivity Analysis: "Often feels stressed"
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    confusion_matrix,
)


# ============================================================
# 1. Paths and settings
# ============================================================

DATA_PATH = "data/processed/mental_health_cleaned.xlsx"
OUTPUT_DIR = "outputs"

RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("SENSITIVITY ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 3. Target columns
# ============================================================

TARGETS = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression",
]

TARGET_MAPPING = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Extremely": 4,
}


# ============================================================
# 4. Feature groups
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
    "Fear of not achieving life goals/ expectations",
]

binary_other_cols = [
    "Family Type",
    "Educational Institute Type",
    "Personality type",
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
    "Job Market Insecurity",
]

ordinal_other_cols = [
    "Socio economic status",
    "Academic Year",
]

nominal_cols = [
    "Gender",
    "Residences Area",
    "Marital status",
    "Various struggle to continue study",
    "Educational qualification",
]

numerical_cols = [
    "Age",
]


# ============================================================
# 5. Verify feature groups
# ============================================================

all_feature_groups = (
    binary_yesno_cols
    + binary_other_cols
    + ordinal_3level_cols
    + ordinal_other_cols
    + nominal_cols
    + numerical_cols
)

if len(all_feature_groups) != 77:
    raise ValueError(
        f"Expected 77 predictors, found {len(all_feature_groups)}"
    )

if len(set(all_feature_groups)) != len(all_feature_groups):
    raise ValueError(
        "Duplicate feature found in feature groups."
    )

missing_features = [
    col for col in all_feature_groups
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Expected feature(s) missing from dataset: {missing_features}"
    )

print(
    f"Feature group verification: PASSED "
    f"({len(all_feature_groups)} predictors)"
)


# ============================================================
# 6. Prepare categorical values safely
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
# 7. Category validation
# ============================================================

print("\nChecking expected categories...")

for col in categorical_columns:

    observed = set(
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    # --------------------------------------------------------
    # Binary Yes/No
    # --------------------------------------------------------

    if col in binary_yesno_cols:

        allowed = {
            "No",
            "Yes",
        }

    # --------------------------------------------------------
    # Binary Other
    # --------------------------------------------------------

    elif col == "Family Type":

        allowed = {
            "Single",
            "Joint",
        }

    elif col == "Educational Institute Type":

        allowed = {
            "Public",
            "Private",
        }

    elif col == "Personality type":

        allowed = {
            "Introvert",
            "Extrovert",
        }

    # --------------------------------------------------------
    # 3-level Ordinal
    # --------------------------------------------------------

    elif col in ordinal_3level_cols:

        allowed = {
            "Never",
            "Sometimes",
            "Always",
        }

    # --------------------------------------------------------
    # Socioeconomic Status
    # --------------------------------------------------------

    elif col == "Socio economic status":

        allowed = {
            "Lower",
            "Lower Middle",
            "Middle",
            "Upper Middle",
            "Upper",
        }

    # --------------------------------------------------------
    # Academic Year
    # --------------------------------------------------------

    elif col == "Academic Year":

        allowed = {
            "1st year",
            "2nd year",
            "3rd year",
            "4th year",
        }

    # --------------------------------------------------------
    # Nominal variables
    # --------------------------------------------------------

    else:
        continue

    unexpected = observed - allowed

    if unexpected:

        raise ValueError(
            f"Unexpected categories in '{col}': "
            f"{unexpected}"
        )

print("Category validation: PASSED")


# ============================================================
# 8. Build parameterized preprocessor
# ============================================================

def build_preprocessor(
    bin_yesno,
    ord_3level,
):

    return ColumnTransformer(
        transformers=[

            # ------------------------------------------------
            # Binary Yes/No
            # ------------------------------------------------

            (
                "binary_yesno",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),

                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[
                                ["No", "Yes"]
                                for _ in bin_yesno
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),

                bin_yesno,
            ),

            # ------------------------------------------------
            # Other Binary Variables
            # ------------------------------------------------

            (
                "binary_other",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),

                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[
                                ["Single", "Joint"],
                                ["Public", "Private"],
                                ["Introvert", "Extrovert"],
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),

                binary_other_cols,
            ),

            # ------------------------------------------------
            # 3-level Ordinal Variables
            # ------------------------------------------------

            (
                "ordinal_3level",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),

                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "Never",
                                    "Sometimes",
                                    "Always",
                                ]
                                for _ in ord_3level
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),

                ord_3level,
            ),

            # ------------------------------------------------
            # Other Ordinal Variables
            # ------------------------------------------------

            (
                "ordinal_other",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),

                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "Lower",
                                    "Lower Middle",
                                    "Middle",
                                    "Upper Middle",
                                    "Upper",
                                ],
                                [
                                    "1st year",
                                    "2nd year",
                                    "3rd year",
                                    "4th year",
                                ],
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),

                ordinal_other_cols,
            ),

            # ------------------------------------------------
            # Nominal Variables
            # ------------------------------------------------

            (
                "nominal",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),

                nominal_cols,
            ),

            # ------------------------------------------------
            # Numerical Variables
            # ------------------------------------------------

            (
                "numerical",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),

                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]),

                numerical_cols,
            ),
        ],

        remainder="drop",

        verbose_feature_names_out=False,
    )


# ============================================================
# 9. Models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# 10. Metric function
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    ordinal_mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    return (
        macro_f1,
        balanced_acc,
        ordinal_mae,
    )


# ============================================================
# 11. Storage
# ============================================================

fold_results = []

class_recall_results = []

confusion_results = []


# ============================================================
# 12. Main Sensitivity Analysis
# ============================================================

for target in TARGETS:

    print("\n" + "=" * 70)
    print(f"TARGET: {target}")
    print("=" * 70)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    y = (
        df[target]
        .map(TARGET_MAPPING)
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Predictor columns
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in all_feature_groups
        if col not in TARGETS
    ]

    X_full = df[feature_columns].copy()

    # --------------------------------------------------------
    # Two experiments
    # --------------------------------------------------------

    experiments = {

        "With_Often_feels_stressed":
            X_full.copy(),

        "Without_Often_feels_stressed":
            X_full.drop(
                columns=["Often feels stressed"]
            ),
    }

    # ========================================================
    # Experiment loop
    # ========================================================

    for experiment_name, X in experiments.items():

        print(
            f"\n--- {experiment_name} ---"
        )

        # ----------------------------------------------------
        # Copy feature groups
        # ----------------------------------------------------

        exp_binary_yesno = (
            binary_yesno_cols.copy()
        )

        exp_ordinal_3level = (
            ordinal_3level_cols.copy()
        )

        # ----------------------------------------------------
        # Remove feature from ordinal group
        # ----------------------------------------------------

        if "Often feels stressed" not in X.columns:

            exp_ordinal_3level.remove(
                "Often feels stressed"
            )

        # ----------------------------------------------------
        # Build preprocessor
        # ----------------------------------------------------

        preprocessor = build_preprocessor(
            exp_binary_yesno,
            exp_ordinal_3level,
        )

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        # ====================================================
        # Model loop
        # ====================================================

        for model_name, model in models.items():

            print(
                f"\nModel: {model_name}"
            )

            # ------------------------------------------------
            # OOF prediction array
            # ------------------------------------------------

            oof_predictions = np.full(
                shape=len(y),
                fill_value=-1,
                dtype=int,
            )

            # =================================================
            # Fold loop
            # =================================================

            for fold, (
                train_idx,
                valid_idx
            ) in enumerate(
                cv.split(X, y),
                start=1,
            ):

                X_train = (
                    X.iloc[train_idx]
                    .copy()
                )

                X_valid = (
                    X.iloc[valid_idx]
                    .copy()
                )

                y_train = y[train_idx]

                y_valid = y[valid_idx]

                # --------------------------------------------
                # Fresh pipeline for every fold
                # --------------------------------------------

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        clone(preprocessor),
                    ),

                    (
                        "model",
                        clone(model),
                    ),
                ])

                # --------------------------------------------
                # Fit ONLY on training fold
                # --------------------------------------------

                pipeline.fit(
                    X_train,
                    y_train,
                )

                # --------------------------------------------
                # Validation prediction
                # --------------------------------------------

                y_pred = pipeline.predict(
                    X_valid
                )

                # --------------------------------------------
                # Store OOF predictions
                # --------------------------------------------

                oof_predictions[
                    valid_idx
                ] = y_pred

                # --------------------------------------------
                # Fold metrics
                # --------------------------------------------

                (
                    macro_f1,
                    balanced_acc,
                    ordinal_mae,
                ) = calculate_metrics(
                    y_valid,
                    y_pred,
                )

                fold_results.append({

                    "target":
                        target,

                    "experiment":
                        experiment_name,

                    "model":
                        model_name,

                    "fold":
                        fold,

                    "macro_f1":
                        macro_f1,

                    "balanced_accuracy":
                        balanced_acc,

                    "ordinal_mae":
                        ordinal_mae,
                })

            # =================================================
            # OOF Integrity Check
            # =================================================

            if np.any(
                oof_predictions == -1
            ):

                raise RuntimeError(
                    "OOF prediction missing for "
                    f"{target} / "
                    f"{experiment_name} / "
                    f"{model_name}"
                )

            # =================================================
            # OOF Confusion Matrix
            # =================================================

            cm = confusion_matrix(
                y,
                oof_predictions,
                labels=[0, 1, 2, 3, 4],
            )

            # -------------------------------------------------
            # Class support
            # -------------------------------------------------

            class_support = (
                cm.sum(axis=1)
            )

            # -------------------------------------------------
            # Class recall
            # -------------------------------------------------

            class_recall = np.divide(
                np.diag(cm),
                class_support,
                out=np.zeros(5),
                where=class_support != 0,
            )

            class_names = [
                "Normal",
                "Mild",
                "Moderate",
                "Severe",
                "Extremely",
            ]

            for class_id, recall in enumerate(
                class_recall
            ):

                class_recall_results.append({

                    "target":
                        target,

                    "experiment":
                        experiment_name,

                    "model":
                        model_name,

                    "class_id":
                        class_id,

                    "class_name":
                        class_names[class_id],

                    "recall":
                        recall,

                    "support":
                        int(
                            class_support[class_id]
                        ),
                })

            # =================================================
            # Store confusion matrix
            # =================================================

            for true_class in range(5):

                for predicted_class in range(5):

                    confusion_results.append({

                        "target":
                            target,

                        "experiment":
                            experiment_name,

                        "model":
                            model_name,

                        "true_class":
                            true_class,

                        "predicted_class":
                            predicted_class,

                        "count":
                            int(
                                cm[
                                    true_class,
                                    predicted_class
                                ]
                            ),
                    })

            # =================================================
            # Overall OOF metrics
            # =================================================

            overall_macro_f1 = f1_score(
                y,
                oof_predictions,
                average="macro",
                zero_division=0,
            )

            overall_balanced_acc = (
                balanced_accuracy_score(
                    y,
                    oof_predictions,
                )
            )

            overall_ordinal_mae = (
                mean_absolute_error(
                    y,
                    oof_predictions,
                )
            )

            print(
                f"  OOF Macro-F1: "
                f"{overall_macro_f1:.4f}"
            )

            print(
                f"  OOF Balanced Accuracy: "
                f"{overall_balanced_acc:.4f}"
            )

            print(
                f"  OOF Ordinal MAE: "
                f"{overall_ordinal_mae:.4f}"
            )


# ============================================================
# 13. Fold-level results
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

fold_output = os.path.join(
    OUTPUT_DIR,
    "sensitivity_fold_results.csv",
)

fold_df.to_csv(
    fold_output,
    index=False,
)


# ============================================================
# 14. Summary statistics
# ============================================================

summary = (
    fold_df
    .groupby(
        [
            "target",
            "experiment",
            "model",
        ]
    )
    .agg(

        macro_f1_mean=(
            "macro_f1",
            "mean",
        ),

        macro_f1_sd=(
            "macro_f1",
            "std",
        ),

        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean",
        ),

        balanced_accuracy_sd=(
            "balanced_accuracy",
            "std",
        ),

        ordinal_mae_mean=(
            "ordinal_mae",
            "mean",
        ),

        ordinal_mae_sd=(
            "ordinal_mae",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# 15. Save summary
# ============================================================

summary_output = os.path.join(
    OUTPUT_DIR,
    "sensitivity_summary.csv",
)

summary.to_csv(
    summary_output,
    index=False,
)


# ============================================================
# 16. Class recall output
# ============================================================

class_recall_df = pd.DataFrame(
    class_recall_results
)

class_recall_output = os.path.join(
    OUTPUT_DIR,
    "sensitivity_class_recall.csv",
)

class_recall_df.to_csv(
    class_recall_output,
    index=False,
)


# ============================================================
# 17. Confusion matrix output
# ============================================================

confusion_df = pd.DataFrame(
    confusion_results
)

confusion_output = os.path.join(
    OUTPUT_DIR,
    "sensitivity_confusion_matrices.csv",
)

confusion_df.to_csv(
    confusion_output,
    index=False,
)


# ============================================================
# 18. Delta Analysis
# ============================================================

pivot_summary = summary.pivot(
    index=[
        "target",
        "model",
    ],
    columns="experiment",
    values=[
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "ordinal_mae_mean",
    ],
)


# ------------------------------------------------------------
# Macro-F1 Delta
# Delta = Without - With
# ------------------------------------------------------------

pivot_summary["macro_f1_delta"] = (

    pivot_summary[
        (
            "macro_f1_mean",
            "Without_Often_feels_stressed",
        )
    ]

    -

    pivot_summary[
        (
            "macro_f1_mean",
            "With_Often_feels_stressed",
        )
    ]
)


# ------------------------------------------------------------
# Balanced Accuracy Delta
# Delta = Without - With
# ------------------------------------------------------------

pivot_summary[
    "balanced_accuracy_delta"
] = (

    pivot_summary[
        (
            "balanced_accuracy_mean",
            "Without_Often_feels_stressed",
        )
    ]

    -

    pivot_summary[
        (
            "balanced_accuracy_mean",
            "With_Often_feels_stressed",
        )
    ]
)


# ------------------------------------------------------------
# Ordinal MAE Delta
# Delta = Without - With
# ------------------------------------------------------------

pivot_summary[
    "ordinal_mae_delta"
] = (

    pivot_summary[
        (
            "ordinal_mae_mean",
            "Without_Often_feels_stressed",
        )
    ]

    -

    pivot_summary[
        (
            "ordinal_mae_mean",
            "With_Often_feels_stressed",
        )
    ]
)


# ============================================================
# 19. Delta table
# ============================================================

delta_table = pivot_summary[
    [
        "macro_f1_delta",
        "balanced_accuracy_delta",
        "ordinal_mae_delta",
    ]
].reset_index()


# ============================================================
# 20. Save delta table
# ============================================================

delta_output = os.path.join(
    OUTPUT_DIR,
    "sensitivity_delta_comparison.csv",
)

delta_table.to_csv(
    delta_output,
    index=False,
)


# ============================================================
# 21. Print final summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL SENSITIVITY SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 22. Print delta comparison
# ============================================================

print("\n" + "=" * 70)
print("SENSITIVITY DELTA (Without - With)")
print("=" * 70)

print(
    delta_table.to_string(
        index=False
    )
)


# ============================================================
# 23. Output files
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"  {fold_output}"
)

print(
    f"  {summary_output}"
)

print(
    f"  {class_recall_output}"
)

print(
    f"  {confusion_output}"
)

print(
    f"  {delta_output}"
)


# ============================================================
# 24. Final status
# ============================================================

print("\nSensitivity analysis completed successfully.")