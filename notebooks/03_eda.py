import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load cleaned dataset ----
df = pd.read_excel('data/processed/mental_health_cleaned.xlsx')

order = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely']
targets = ['DASS_Stress', 'DASS-Anxiety', 'DASS-Depression']

# ==========================================
# PART A: Target Distribution Charts
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, target in enumerate(targets):
    sns.countplot(data=df, x=target, order=order, ax=axes[i], palette='viridis')
    axes[i].set_title(f'{target} Distribution')
    axes[i].set_xlabel('Severity Level')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/target_distributions.png', dpi=300)
print("Chart saved to outputs/target_distributions.png")
plt.close()  # show() ki jagah close() use kiya taaki script bina ruke aage badhe

# ==========================================
# PART B: Label Validity Check (inter-target consistency)
# ==========================================
# Ordinal categories ko 0-4 codes mein convert kiya (Normal=0 ... Extremely=4)
for col in targets:
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)

df_numeric = df[targets].apply(lambda x: x.cat.codes)

print("\n--- Inter-target Spearman correlation ---")
print(df_numeric.corr(method='spearman'))

# ==========================================
# PART C: Self-report consistency check
# ==========================================
from scipy.stats import spearmanr

stress_map = {'Never': 0, 'Sometimes': 1, 'Always': 2}
self_report_numeric = df['Often feels stressed'].map(stress_map)
dass_stress_numeric = df['DASS_Stress'].cat.codes

corr, pval = spearmanr(self_report_numeric, dass_stress_numeric)
print(f"\n--- Often feels stressed vs DASS_Stress ---")
print(f"Spearman correlation: {corr:.3f}, p-value: {pval:.5f}")

print("\n--- Crosstab: Often feels stressed vs DASS_Stress ---")
print(pd.crosstab(df['Often feels stressed'], df['DASS_Stress']))

# ==========================================
# PART D: Correlation heatmap (bonus visual for paper)
# ==========================================
plt.figure(figsize=(6, 5))
sns.heatmap(df_numeric.corr(method='spearman'), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Inter-target Spearman Correlation')
plt.tight_layout()
plt.savefig('outputs/target_correlation_heatmap.png', dpi=300)
print("\nHeatmap saved to outputs/target_correlation_heatmap.png")
plt.close()


print("\n--- Gender vs Mental Health Targets ---")

for target in ["DASS_Stress", "DASS-Anxiety", "DASS-Depression"]:
    print(f"\n{target}")
    print(pd.crosstab(df["Gender"], df[target], normalize="index").round(3))


print("\n--- Family Type vs Mental Health Targets ---")

for target in targets:
    print(f"\n{target}")
    print(pd.crosstab(df["Family Type"], df[target], normalize="index").round(3))


print("\n--- Academic Year vs Mental Health Targets ---")

for target in targets:
    print(f"\n{target}")
    print(pd.crosstab(df["Academic Year"], df[target], normalize="index").round(3))


# ==========================================
# PART E: Categorical Features vs DASS Targets
# ==========================================

# Features ko exclude karna hai jo targets hain
categorical_features = [
    col for col in df.select_dtypes(include=["object", "string", "category"]).columns
    if col not in targets
]

print("\n==========================================")
print("Categorical Features vs DASS Targets")
print("==========================================")

for feature in categorical_features:

    print(f"\n\n### {feature} ###")

    for target in targets:

        print(f"\n{target}")

        table = pd.crosstab(
            df[feature],
            df[target],
            normalize="index"
        ).round(3)

        print(table)


