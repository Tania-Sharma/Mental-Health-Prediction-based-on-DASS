import pandas as pd
from pathlib import Path


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel("data/processed/mental_health_cleaned.xlsx")

targets = [
    "DASS_Stress",
    "DASS-Anxiety",
    "DASS-Depression"
]


# ==========================================
# SEPARATE PREDICTORS AND TARGETS
# ==========================================

# Timestamp aur Educational Institute Name
# already processed dataset se remove ho chuke hain.
# Isliye yahan sirf target columns remove kar rahe hain.

X = df.drop(columns=targets)
y = df[targets].copy()

print("Predictors after exclusions:", X.shape[1])


# ==========================================
# COLUMN CLASSIFICATION
# ==========================================

# ------------------------------------------
# 1. Binary Yes/No columns
# ------------------------------------------

binary_yesno_cols = [
    'Taking counseling',
    'Mental illness history in the family',
    'Death in the family',
    'Frequent arguments among parents',
    'Divorced family',
    'Too strict guardians',
    'Frequent conflicts with family members',
    'Addicted family member',
    'Traumatic childhood experience',
    'Quality relationship with family',
    'Emotionally supportive family',
    'Satisfied relationship with family',
    'Biased parents',
    'Family pressure on academic selection',
    'Frequent failure in exam',
    'Worried about academic failure',
    'Family pressure on performing well in exam',
    'Parental satisfaction with academic result',
    'Self-satisfaction with academic performances',
    'Pressure of academic competition',
    'Breakup of a romantic relationship',
    'Insecurity of physical appearance',
    'Fear of communicating with new people',
    'Lack of money to meet basic needs',
    'Difficulty in maintaining a standard lifestyle',
    'Prefer to be alone mostly',
    'Social media monitoring',
    'Social media post review',
    'Unhappy feelings from social media',
    'Lack of personal space at home/ hostel',
    'Unsafe Residence',
    'Dissatisfied Neighborhood',
    'Experience social restrictions',
    'Dissatisfaction with living environment',
    'Physical abuse',
    'Verbal violence/ abuse',
    'Emotional violence/ abuse',
    'Sexual abuse',
    'Social violence/ abuse',
    'Pressure to meet family expectations',
    'Not achieving target/ expectation.',
    'Fear of not achieving life goals/ expectations'
]


# ------------------------------------------
# 2. Binary categorical columns
# ------------------------------------------

binary_other_cols = {
    'Family Type': {
        'Single': 0,
        'Joint': 1
    },

    'Educational Institute Type': {
        'Public': 0,
        'Private': 1
    },

    'Personality type': {
        'Introvert': 0,
        'Extrovert': 1
    }
}


# ------------------------------------------
# 3. Ordinal 3-level columns
# ------------------------------------------

ordinal_3level_cols = [
    'Participation in extracurricular activities',
    'Play video-game',
    'Participate in any sports.',
    'Smoking cigarette',
    'Smoking of marijuana/weed/ alcohol',
    'Taking drugs or any other substance?',
    'Falling sick frequently',
    'Maintaining a balanced diet',
    'Often sleep late at night.',
    'Physical exercise/ Yoga',
    'Religious practices',
    'Often arguments/fights with friends',
    'Bullied/ verbally abused by friends/ others',
    'Cheated by friend',
    'Abused/cheated by partner',
    'Supportive friends',
    'Biased Teacher',
    'Insulted/ harassed by teacher',
    'Inferiority in friendship',
    'Often feels stressed',
    'Deprived of deservation',
    'Inferiority complex',
    'Time spent on social media',
    'Job Market Insecurity'
]


# ------------------------------------------
# 4. Other ordinal columns
# ------------------------------------------

ordinal_other_cols = {
    'Socio economic status': [
        'Lower',
        'Lower Middle',
        'Middle',
        'Upper Middle',
        'Upper'
    ],

    'Academic Year': [
        '1st year',
        '2nd year',
        '3rd year',
        '4th year'
    ]
}


# ------------------------------------------
# 5. Nominal categorical columns
# ------------------------------------------

nominal_cols = [
    'Gender',
    'Residences Area',
    'Marital status',
    'Various struggle to continue study',
    'Educational qualification'
]


# ------------------------------------------
# 6. Numerical columns
# ------------------------------------------

numerical_cols = [
    'Age'
]


# ==========================================
# SANITY CHECK
# ==========================================

classified = (
    set(binary_yesno_cols)
    | set(binary_other_cols)
    | set(ordinal_3level_cols)
    | set(ordinal_other_cols)
    | set(nominal_cols)
    | set(numerical_cols)
)

print("\n==========================================")
print("COLUMN CLASSIFICATION CHECK")
print("==========================================")

print(
    "Unclassified columns (should be empty):",
    set(X.columns) - classified
)

print(
    "Classified but missing from data (should be empty):",
    classified - set(X.columns)
)


# ==========================================
# APPLY ENCODINGS
# ==========================================

X_encoded = pd.DataFrame(index=X.index)


# ------------------------------------------
# 1. Binary Yes/No -> 0/1
# ------------------------------------------

for col in binary_yesno_cols:
    X_encoded[col] = X[col].map({
        'No': 0,
        'Yes': 1
    })


# ------------------------------------------
# 2. Binary categorical -> 0/1
# ------------------------------------------

for col, mapping in binary_other_cols.items():
    X_encoded[col] = X[col].map(mapping)


# ------------------------------------------
# 3. Ordinal 3-level -> 0/1/2
# ------------------------------------------

freq_map = {
    'Never': 0,
    'Sometimes': 1,
    'Always': 2
}

for col in ordinal_3level_cols:
    X_encoded[col] = X[col].map(freq_map)


# ------------------------------------------
# 4. Other ordinal columns -> ordered integers
# ------------------------------------------

for col, order in ordinal_other_cols.items():

    mapping = {
        category: i
        for i, category in enumerate(order)
    }

    X_encoded[col] = X[col].map(mapping)


# ------------------------------------------
# 5. Numerical columns -> unchanged
# ------------------------------------------

for col in numerical_cols:
    X_encoded[col] = X[col]


# ------------------------------------------
# 6. Nominal columns -> One-Hot Encoding
# ------------------------------------------

nominal_encoded = pd.get_dummies(
    X[nominal_cols],
    prefix=nominal_cols,
    dtype=int
)


# ==========================================
# COMBINE ALL ENCODED FEATURES
# ==========================================

X_final = pd.concat(
    [
        X_encoded,
        nominal_encoded
    ],
    axis=1
)


# ==========================================
# ENCODING RESULT
# ==========================================

print("\n==========================================")
print("ENCODING RESULT")
print("==========================================")

print("Original predictor columns:", X.shape[1])
print("Final encoded shape:", X_final.shape)

print(
    "Missing values after encoding:",
    X_final.isna().sum().sum()
)


# ==========================================
# SAVE OUTPUT
# ==========================================

Path("data/processed").mkdir(
    exist_ok=True
)

X_final.to_csv(
    "data/processed/mental_health_encoded_v2.csv",
    index=False
)

y.to_csv(
    "data/processed/mental_health_targets.csv",
    index=False
)


print("\n==========================================")
print("FILES SAVED")
print("==========================================")

print(
    "Saved: data/processed/mental_health_encoded_v2.csv"
)

print(
    "Saved: data/processed/mental_health_targets.csv"
)