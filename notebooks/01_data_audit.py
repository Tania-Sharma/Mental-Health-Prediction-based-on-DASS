import pandas as pd

# ---- Load dataset ----
file_path = 'data/raw/mental_health.xlsx'
df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

pd.set_option('display.max_rows', None)  # sabkuch dikhega, truncate nahi hoga

# ---- Data types ----
print("\n--- Full dtypes ---")
print(df.dtypes)

print("\n--- Dtype summary ---")
print(df.dtypes.value_counts())

# ---- Missing values ----
print("\n--- Columns with missing values (non-zero only) ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Total missing values ---")
print(df.isnull().sum().sum())

# ---- Duplicates ----
print("\n--- Duplicate rows ---")
print("Number of duplicate rows:", df.duplicated().sum())

# ---- Age column check ----
print("\n--- Age column unique values ---")
print(df['Age '].unique())   # note: original column name has trailing space

# ---- Target columns check ----
print("\n--- Target columns unique values ---")
print("DASS_Stress:", df['DASS_Stress'].unique())
print("DASS-Anxiety:", df['DASS-Anxiety'].unique())
print("DASS-Depression:", df['DASS-Depression'].unique())

# ---- Target distributions ----
print("\n--- Target distribution: DASS_Stress ---")
print(df['DASS_Stress'].value_counts())

print("\n--- Target distribution: DASS-Anxiety ---")
print(df['DASS-Anxiety'].value_counts())

print("\n--- Target distribution: DASS-Depression ---")
print(df['DASS-Depression'].value_counts())

# ---- Sample of categorical columns to understand encoding patterns ----
print("\n--- Unique values for a sample of columns from each category ---")
sample_cols = [
    'Gender', 'Residences Area', 'Family Type', 'Socio economic status',
    'Mental illness history in the family', 'Frequent arguments among parents',
    'Quality relationship with family', 'Frequent failure in exam',
    'Participation in extracurricular activities', 'Play video-game',
    'Smoking cigarette', 'Time spent on social media ',
    'Physical abuse ', 'Personality type', 'Academic Year'
]
for col in sample_cols:
    print(f"\n{col}: {df[col].unique()}")