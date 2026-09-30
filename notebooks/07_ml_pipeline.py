import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/processed/mental_health_cleaned.xlsx"

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


# ============================================================
# 2. TARGETS
# ============================================================

targets = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression"
]

target_mapping = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Extremely": 4
}

y = df[targets].copy()

for target in targets:

    y[target] = y[target].map(target_mapping)

    if y[target].isna().any():
        raise ValueError(
            f"Unexpected or missing target category found in: {target}"
        )


# ============================================================
# 3. CREATE PREDICTOR DATA
# ============================================================

X = df.drop(columns=targets).copy()


# ============================================================
# 4. IDENTIFIER / NON-PREDICTIVE COLUMN CHECK
# ============================================================

identifier_columns = [
    "Timestamp",
    "Educational Institute Name"
]

remaining_identifiers = [
    col for col in identifier_columns
    if col in X.columns
]

if remaining_identifiers:

    raise ValueError(
        "\nIdentifier columns are still present in the dataset:\n"
        f"{remaining_identifiers}\n\n"
        "Remove these columns from the cleaned dataset before "
        "continuing."
    )


# ============================================================
# 5. FEATURE GROUPS
# ============================================================


# ------------------------------------------------------------
# 5.1 Binary Yes / No
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5.2 Binary categorical variables
# ------------------------------------------------------------

binary_other_cols = [

    "Family Type",
    "Educational Institute Type",
    "Personality type"
]

binary_other_categories = {

    "Family Type": [
        "Single",
        "Joint"
    ],

    "Educational Institute Type": [
        "Public",
        "Private"
    ],

    "Personality type": [
        "Introvert",
        "Extrovert"
    ]
}


# ------------------------------------------------------------
# 5.3 Ordinal: Never < Sometimes < Always
# ------------------------------------------------------------

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

ordinal_3level_categories = [
    [
        "Never",
        "Sometimes",
        "Always"
    ]
    for _ in ordinal_3level_cols
]


# ------------------------------------------------------------
# 5.4 Other ordinal variables
# ------------------------------------------------------------

ordinal_other_cols = [

    "Socio economic status",
    "Academic Year"
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


# ------------------------------------------------------------
# 5.5 Nominal categorical variables
# ------------------------------------------------------------

nominal_cols = [

    "Gender",
    "Residences Area",
    "Marital status",
    "Various struggle to continue study",
    "Educational qualification"
]


# ------------------------------------------------------------
# 5.6 Numerical variables
# ------------------------------------------------------------

numerical_cols = [
    "Age"
]


# ============================================================
# 6. VERIFY FEATURE GROUPS
# ============================================================

all_grouped_features = (

    binary_yesno_cols
    + binary_other_cols
    + ordinal_3level_cols
    + ordinal_other_cols
    + nominal_cols
    + numerical_cols
)


# Check duplicate assignment
feature_series = pd.Series(all_grouped_features)

duplicate_features = (
    feature_series[
        feature_series.duplicated()
    ]
    .unique()
)


if len(duplicate_features) > 0:

    raise ValueError(
        "\nFeature assigned to more than one group:\n"
        f"{duplicate_features}"
    )


# Check missing columns
missing_grouped_columns = [

    col
    for col in all_grouped_features
    if col not in X.columns
]


if missing_grouped_columns:

    raise ValueError(
        "\nThese expected predictor columns are missing:\n"
        f"{missing_grouped_columns}"
    )


# Check unclassified columns
unclassified_columns = (

    set(X.columns)
    - set(all_grouped_features)
)


if unclassified_columns:

    raise ValueError(
        "\nUnclassified predictor columns found:\n"
        f"{sorted(unclassified_columns)}"
    )


print("\nFeature-group verification: PASSED")

print(
    "Binary Yes/No:",
    len(binary_yesno_cols)
)

print(
    "Binary other:",
    len(binary_other_cols)
)

print(
    "Ordinal 3-level:",
    len(ordinal_3level_cols)
)

print(
    "Ordinal other:",
    len(ordinal_other_cols)
)

print(
    "Nominal:",
    len(nominal_cols)
)

print(
    "Numerical:",
    len(numerical_cols)
)

print(
    "Total predictors:",
    len(all_grouped_features)
)


# ============================================================
# 7. STANDARDIZE CATEGORY TEXT SPACING
# ============================================================

categorical_columns = (

    binary_yesno_cols
    + binary_other_cols
    + ordinal_3level_cols
    + ordinal_other_cols
    + nominal_cols
)


for col in categorical_columns:

    X[col] = (
        X[col]
        .astype("object")
        .apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else np.nan
            if pd.isna(value)
            else value
        )
    )


# ============================================================
# 8. CATEGORY VALIDATION FUNCTION
# ============================================================

def validate_categories(data, expected_categories):

    for column, allowed_categories in expected_categories.items():

        actual_categories = set(

            data[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()

        )

        unexpected_categories = (

            actual_categories
            - set(allowed_categories)

        )

        if unexpected_categories:

            raise ValueError(

                f"\nUnexpected category found in '{column}':\n"
                f"Found: {sorted(unexpected_categories)}\n"
                f"Expected: {allowed_categories}"

            )


# ============================================================
# 9. VALIDATE ALL EXPLICIT CATEGORIES
# ============================================================


# Yes / No variables

yesno_expected = {

    col: [
        "No",
        "Yes"
    ]

    for col in binary_yesno_cols
}


validate_categories(
    X,
    yesno_expected
)


# Binary categorical variables

validate_categories(
    X,
    binary_other_categories
)


# Never / Sometimes / Always variables

ordinal_expected = {

    col: [
        "Never",
        "Sometimes",
        "Always"
    ]

    for col in ordinal_3level_cols
}


validate_categories(
    X,
    ordinal_expected
)


# Socioeconomic status + Academic Year

validate_categories(
    X,
    {
        "Socio economic status": [
            "Lower",
            "Lower Middle",
            "Middle",
            "Upper Middle",
            "Upper"
        ],

        "Academic Year": [
            "1st year",
            "2nd year",
            "3rd year",
            "4th year"
        ]
    }
)


print(
    "\nCategory validation: PASSED"
)


# ============================================================
# 10. PREPROCESSING PIPELINES
# ============================================================


# ------------------------------------------------------------
# 10.1 Binary Yes / No
# ------------------------------------------------------------

binary_pipeline = Pipeline(
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

                categories=[
                    [
                        "No",
                        "Yes"
                    ]

                    for _ in binary_yesno_cols
                ],

                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 10.2 Binary categorical variables
# ------------------------------------------------------------

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

                categories=[
                    binary_other_categories[col]
                    for col in binary_other_cols
                ],

                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 10.3 Three-level ordinal variables
# ------------------------------------------------------------

ordinal_pipeline = Pipeline(
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


# ------------------------------------------------------------
# 10.4 Other ordinal variables
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 10.5 Nominal variables
# ------------------------------------------------------------

nominal_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# 10.6 Numerical variables
# ------------------------------------------------------------

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


# ============================================================
# 11. COMPLETE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "binary_yesno",
            binary_pipeline,
            binary_yesno_cols
        ),

        (
            "binary_other",
            binary_other_pipeline,
            binary_other_cols
        ),

        (
            "ordinal_3level",
            ordinal_pipeline,
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
# 12. CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


# ============================================================
# 13. LEAKAGE-SAFE CROSS-VALIDATION TEST
# ============================================================

summary_rows = []

feature_name_rows = []


for target in targets:

    print("\n")
    print("=" * 65)
    print(f"TARGET: {target}")
    print("=" * 65)

    target_y = y[target]

    fold_number = 1


    for train_idx, valid_idx in cv.split(
        X,
        target_y
    ):

        print("\n" + "-" * 65)
        print(f"Fold {fold_number}")
        print("-" * 65)


        # ----------------------------------------------------
        # Split data
        # ----------------------------------------------------

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        y_train = target_y.iloc[train_idx]
        y_valid = target_y.iloc[valid_idx]


        print(
            "Training samples:",
            len(X_train)
        )

        print(
            "Validation samples:",
            len(X_valid)
        )


        # ----------------------------------------------------
        # Check target classes
        # ----------------------------------------------------

        train_classes = sorted(
            y_train.unique()
        )

        valid_classes = sorted(
            y_valid.unique()
        )


        print(
            "Training classes:",
            train_classes
        )

        print(
            "Validation classes:",
            valid_classes
        )


        expected_classes = [
            0,
            1,
            2,
            3,
            4
        ]


        if train_classes != expected_classes:

            raise ValueError(

                f"Training fold {fold_number} "
                f"for {target} is missing a class: "
                f"{train_classes}"

            )


        if valid_classes != expected_classes:

            raise ValueError(

                f"Validation fold {fold_number} "
                f"for {target} is missing a class: "
                f"{valid_classes}"

            )


        # ----------------------------------------------------
        # CRITICAL:
        # Fresh preprocessor for every fold
        # ----------------------------------------------------

        fold_preprocessor = clone(
            preprocessor
        )


        # ----------------------------------------------------
        # FIT ONLY ON TRAINING DATA
        # ----------------------------------------------------

        X_train_transformed = fold_preprocessor.fit_transform(X_train)


        # ----------------------------------------------------
        # TRANSFORM VALIDATION DATA
        # WITHOUT FITTING
        # ----------------------------------------------------

        X_valid_transformed = (

            fold_preprocessor
            .transform(
                X_valid
            )

        )


        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        print(
            "Transformed training shape:",
            X_train_transformed.shape
        )

        print(
            "Transformed validation shape:",
            X_valid_transformed.shape
        )


        if (
            X_train_transformed.shape[1]
            != X_valid_transformed.shape[1]
        ):

            raise ValueError(
                "Training and validation transformed "
                "dimensions do not match."
            )


        # ----------------------------------------------------
        # Check NaN
        # ----------------------------------------------------

        if np.isnan(
            X_train_transformed
        ).any():

            raise ValueError(
                "NaN found in transformed training data."
            )


        if np.isnan(
            X_valid_transformed
        ).any():

            raise ValueError(
                "NaN found in transformed validation data."
            )


        # ----------------------------------------------------
        # Check infinite values
        # ----------------------------------------------------

        if not np.isfinite(
            X_train_transformed
        ).all():

            raise ValueError(
                "Non-finite value found "
                "in transformed training data."
            )


        if not np.isfinite(
            X_valid_transformed
        ).all():

            raise ValueError(
                "Non-finite value found "
                "in transformed validation data."
            )


        # ----------------------------------------------------
        # Check unknown ordinal categories
        # ----------------------------------------------------
        #
        # Unknown ordinal categories are encoded as -1.
        # Because we validated the complete dataset above,
        # this should normally remain zero.
        #

        unknown_train = np.sum(
            X_train_transformed == -1
        )

        unknown_valid = np.sum(
            X_valid_transformed == -1
        )


        print(
            "Unknown encoded values in training:",
            unknown_train
        )

        print(
            "Unknown encoded values in validation:",
            unknown_valid
        )


        # ----------------------------------------------------
        # Get feature names from this fold
        # ----------------------------------------------------

        try:

            feature_names = (
                fold_preprocessor
                .get_feature_names_out()
            )

            print(
                "Number of feature names:",
                len(feature_names)
            )

        except Exception:

            feature_names = []


        # Save feature names only once per target/fold
        feature_name_rows.append({

            "target": target,

            "fold": fold_number,

            "number_of_features":
                X_train_transformed.shape[1],

            "feature_names":
                " | ".join(feature_names)

        })


        # ----------------------------------------------------
        # Save fold summary
        # ----------------------------------------------------

        summary_rows.append({

            "target": target,

            "fold": fold_number,

            "train_samples":
                len(train_idx),

            "validation_samples":
                len(valid_idx),

            "train_classes":
                str(train_classes),

            "validation_classes":
                str(valid_classes),

            "transformed_features":
                X_train_transformed.shape[1],

            "unknown_train_values":
                int(unknown_train),

            "unknown_validation_values":
                int(unknown_valid)

        })


        fold_number += 1


# ============================================================
# 14. SAVE PIPELINE VALIDATION SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_output = (
    "outputs/ml_pipeline_summary_corrected.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


# ============================================================
# 15. SAVE FEATURE NAMES
# ============================================================

feature_names_df = pd.DataFrame(
    feature_name_rows
)

feature_names_output = (
    "outputs/ml_pipeline_feature_names.csv"
)

feature_names_df.to_csv(
    feature_names_output,
    index=False
)


# ============================================================
# 16. FINAL CHECKS
# ============================================================

print("\n")
print("=" * 65)
print("CORRECTED ML PIPELINE VALIDATION COMPLETE")
print("=" * 65)


print(
    "\nOriginal predictor missing values:",
    X.isna().sum().sum()
)


print(
    "\nSummary:"
)

print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    summary_output
)

print(
    feature_names_output
)


print(
    "\nAll pipeline checks PASSED."
)