import pandas as pd


# ==========================================
# 1. LOAD RAW DATA
# ==========================================

file_path = "data/raw/mental_health.xlsx"

df = pd.read_excel(file_path)


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

rename_map = {
    "Emotionally supportivefamily":
        "Emotionally supportive family",

    "Physicalexercise/ Yoga":
        "Physical exercise/ Yoga",

    "Breakupof a romantic relationship":
        "Breakup of a romantic relationship"
}

df.rename(columns=rename_map, inplace=True)


# ==========================================
# 3. REMOVE NON-PREDICTIVE IDENTIFIERS
# ==========================================

drop_columns = [
    "Timestamp",
    "Educational Institute Name"
]

existing_drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

df.drop(
    columns=existing_drop_columns,
    inplace=True
)

print("\nRemoved columns:")
print(existing_drop_columns)


# ==========================================
# 4. CLEAN AGE
# ==========================================

def clean_age(val):

    val = str(val).strip()

    if val == "18-01-2005":
        return 18

    if "+" in val:
        return int(
            val.replace("+", "").strip()
        )

    if "year" in val.lower():
        return int(
            "".join(filter(str.isdigit, val))
        )

    return int(float(val))


df["Age"] = df["Age"].apply(clean_age)

print("\nAge cleaned. Unique values:")
print(sorted(df["Age"].unique()))


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

print("\nMissing values before ML preprocessing:")

missing = df.isnull().sum()

print(
    missing[missing > 0]
)

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)


# ==========================================
# 6. SAVE CLEANED DATA
# ==========================================

df.to_excel(
    "data/processed/mental_health_cleaned.xlsx",
    index=False
)

print(
    "\nFinal shape:",
    df.shape
)

print(
    "\nSaved cleaned dataset to "
    "data/processed/mental_health_cleaned.xlsx"
)