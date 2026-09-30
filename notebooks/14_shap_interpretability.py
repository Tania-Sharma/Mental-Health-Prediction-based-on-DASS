# ============================================================
# 14_shap_interpretability.py
# Model Explainability & Domain Feature Analysis with SHAP
# - Explains final production models for Stress, Anxiety, Depression
# - Uses LinearExplainer for Logistic Regression and TreeExplainer for Random Forest
# - Aggregates feature importances globally and across domain groups
# - Generates publication-ready figures and structured CSVs
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


# ============================================================
# 1. PATHS
# ============================================================

INPUT_FILE = "data/processed/mental_health_cleaned.xlsx"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FEATURE GROUPS & DOMAIN TAXONOMY
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

# 6 High-Level Conceptual Domains
DOMAIN_MAPPING = {
    # 1. Academic & Educational
    "Family pressure on academic selection": "Academic & Institutional",
    "Frequent failure in exam": "Academic & Institutional",
    "Worried about academic failure": "Academic & Institutional",
    "Family pressure on performing well in exam": "Academic & Institutional",
    "Parental satisfaction with academic result": "Academic & Institutional",
    "Self-satisfaction with academic performances": "Academic & Institutional",
    "Pressure of academic competition": "Academic & Institutional",
    "Biased Teacher": "Academic & Institutional",
    "Insulted/ harassed by teacher": "Academic & Institutional",
    "Educational Institute Type": "Academic & Institutional",
    "Academic Year": "Academic & Institutional",
    "Various struggle to continue study": "Academic & Institutional",
    "Educational qualification": "Academic & Institutional",

    # 2. Family & Domestic Environment
    "Mental illness history in the family": "Family & Domestic Environment",
    "Death in the family": "Family & Domestic Environment",
    "Frequent arguments among parents": "Family & Domestic Environment",
    "Divorced family": "Family & Domestic Environment",
    "Too strict guardians": "Family & Domestic Environment",
    "Frequent conflicts with family members": "Family & Domestic Environment",
    "Addicted family member": "Family & Domestic Environment",
    "Traumatic childhood experience": "Family & Domestic Environment",
    "Quality relationship with family": "Family & Domestic Environment",
    "Emotionally supportive family": "Family & Domestic Environment",
    "Satisfied relationship with family": "Family & Domestic Environment",
    "Biased parents": "Family & Domestic Environment",
    "Pressure to meet family expectations": "Family & Domestic Environment",
    "Family Type": "Family & Domestic Environment",

    # 3. Social, Peer & Interpersonal Relations
    "Breakup of a romantic relationship": "Social & Interpersonal",
    "Insecurity of physical appearance": "Social & Interpersonal",
    "Fear of communicating with new people": "Social & Interpersonal",
    "Prefer to be alone mostly": "Social & Interpersonal",
    "Often arguments/fights with friends": "Social & Interpersonal",
    "Bullied/ verbally abused by friends/ others": "Social & Interpersonal",
    "Cheated by friend": "Social & Interpersonal",
    "Abused/cheated by partner": "Social & Interpersonal",
    "Supportive friends": "Social & Interpersonal",
    "Inferiority in friendship": "Social & Interpersonal",
    "Social media monitoring": "Social & Interpersonal",
    "Social media post review": "Social & Interpersonal",
    "Unhappy feelings from social media": "Social & Interpersonal",
    "Time spent on social media": "Social & Interpersonal",
    "Marital status": "Social & Interpersonal",
    "Personality type": "Social & Interpersonal",

    # 4. Lifestyle, Behavior & Substance Use
    "Participation in extracurricular activities": "Lifestyle & Health Behavior",
    "Play video-game": "Lifestyle & Health Behavior",
    "Participate in any sports.": "Lifestyle & Health Behavior",
    "Smoking cigarette": "Lifestyle & Health Behavior",
    "Smoking of marijuana/weed/ alcohol": "Lifestyle & Health Behavior",
    "Taking drugs or any other substance?": "Lifestyle & Health Behavior",
    "Falling sick frequently": "Lifestyle & Health Behavior",
    "Maintaining a balanced diet": "Lifestyle & Health Behavior",
    "Often sleep late at night.": "Lifestyle & Health Behavior",
    "Physical exercise/ Yoga": "Lifestyle & Health Behavior",
    "Religious practices": "Lifestyle & Health Behavior",

    # 5. Trauma, Abuse & Psychological Vulnerability
    "Physical abuse": "Trauma, Abuse & Self-Perception",
    "Verbal violence/ abuse": "Trauma, Abuse & Self-Perception",
    "Emotional violence/ abuse": "Trauma, Abuse & Self-Perception",
    "Sexual abuse": "Trauma, Abuse & Self-Perception",
    "Social violence/ abuse": "Trauma, Abuse & Self-Perception",
    "Deprived of deservation": "Trauma, Abuse & Self-Perception",
    "Inferiority complex": "Trauma, Abuse & Self-Perception",
    "Taking counseling": "Trauma, Abuse & Self-Perception",

    # 6. Economic, Future & Living Conditions
    "Lack of money to meet basic needs": "Economic & Environmental Security",
    "Difficulty in maintaining a standard lifestyle": "Economic & Environmental Security",
    "Lack of personal space at home/ hostel": "Economic & Environmental Security",
    "Unsafe Residence": "Economic & Environmental Security",
    "Dissatisfied Neighborhood": "Economic & Environmental Security",
    "Experience social restrictions": "Economic & Environmental Security",
    "Dissatisfaction with living environment": "Economic & Environmental Security",
    "Job Market Insecurity": "Economic & Environmental Security",
    "Not achieving target/ expectation.": "Economic & Environmental Security",
    "Fear of not achieving life goals/ expectations": "Economic & Environmental Security",
    "Socio economic status": "Economic & Environmental Security",
    "Residences Area": "Economic & Environmental Security",
    "Age": "Demographics",
    "Gender": "Demographics"
}


# ============================================================
# 3. LOAD DATA & FITTED MODELS
# ============================================================

print("=" * 75)
print("STAGE 14: SHAP INTERPRETABILITY & DOMAIN ANALYSIS")
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

X = df[predictor_groups].copy()

targets_config = {
    "DASS_Stress": {
        "file": "final_model_DASS_Stress.joblib",
        "type": "linear",
        "title": "DASS Stress (Regularized Logistic Regression)"
    },
    "DASS-Anxiety": {
        "file": "final_model_DASS_Anxiety.joblib",
        "type": "linear",
        "title": "DASS Anxiety (Regularized Logistic Regression)"
    },
    "DASS-Depression": {
        "file": "final_model_DASS_Depression.joblib",
        "type": "tree",
        "title": "DASS Depression (Random Forest Classifier)"
    }
}


# ============================================================
# 4. COMPUTE SHAP VALUES PER TARGET
# ============================================================

domain_records = []

for target_name, cfg in targets_config.items():
    print(f"\nExplaining model for {target_name} ({cfg['type']})...")

    model_path = os.path.join(MODEL_DIR, cfg["file"])
    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["model"]

    # Transform raw predictors
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Clean feature names (remove transformer prefixes like binary_yesno__)
    clean_feature_names = [
        name.split("__")[-1] for name in feature_names
    ]

    # Map each transformed feature to its base variable name & domain
    feature_to_domain = {}
    for feat in clean_feature_names:
        # Check direct match
        matched_base = None
        for base_col in predictor_groups:
            if feat == base_col or feat.startswith(base_col + "_"):
                matched_base = base_col
                break
        if matched_base is not None:
            domain = DOMAIN_MAPPING.get(matched_base, "Other")
        else:
            domain = "Other"
        feature_to_domain[feat] = (matched_base or feat, domain)

    # Compute SHAP
    if cfg["type"] == "linear":
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        # shap_values shape: (n_samples, n_features, n_classes) or list of (n_samples, n_features)
    elif cfg["type"] == "tree":
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)

    # Calculate global feature importance (mean absolute SHAP across all samples and classes)
    if isinstance(shap_values, list):
        # List of arrays per class: average across classes
        class_shap_means = [np.abs(sv).mean(axis=0) for sv in shap_values]
        global_shap_importance = np.mean(class_shap_means, axis=0)
    elif len(shap_values.shape) == 3:
        # Array of shape (samples, features, classes)
        global_shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        global_shap_importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Transformed_Feature": clean_feature_names,
        "Base_Feature": [feature_to_domain[f][0] for f in clean_feature_names],
        "Domain": [feature_to_domain[f][1] for f in clean_feature_names],
        "Mean_Absolute_SHAP": global_shap_importance
    })

    # Aggregate by Base_Feature (combining one-hot indicator columns for nominal variables)
    base_importance_df = (
        importance_df.groupby(["Base_Feature", "Domain"])["Mean_Absolute_SHAP"]
        .sum()
        .reset_index()
        .sort_values(by="Mean_Absolute_SHAP", ascending=False)
    )

    clean_target = target_name.replace("-", "_")
    csv_out = os.path.join(OUTPUT_DIR, f"shap_importance_{clean_target}.csv")
    base_importance_df.to_csv(csv_out, index=False)
    print(f"Saved feature importances to {csv_out}")

    # Accumulate domain contributions
    for _, row in base_importance_df.iterrows():
        domain_records.append({
            "Target": target_name,
            "Domain": row["Domain"],
            "Base_Feature": row["Base_Feature"],
            "Importance": row["Mean_Absolute_SHAP"]
        })

    # ========================================================
    # Plot Top 15 Predictors Barplot
    # ========================================================
    plt.figure(figsize=(10, 6.5))
    top15 = base_importance_df.head(15).sort_values(by="Mean_Absolute_SHAP", ascending=True)

    # Color palette based on Domain
    palette = sns.color_palette("muted", n_colors=len(top15["Domain"].unique()))
    domain_color_map = dict(zip(top15["Domain"].unique(), palette))
    bar_colors = [domain_color_map[d] for d in top15["Domain"]]

    bars = plt.barh(top15["Base_Feature"], top15["Mean_Absolute_SHAP"], color=bar_colors, edgecolor="#333333", height=0.65)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Output)", fontsize=11, fontweight="bold")
    plt.title(f"Top 15 Predictors of {cfg['title']}", fontsize=13, fontweight="bold", pad=12)

    # Add legend for domains
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="#333333", label=d) for d, c in domain_color_map.items()]
    plt.legend(handles=legend_elements, title="Domain", loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig_out = os.path.join(OUTPUT_DIR, f"shap_summary_{clean_target}.png")
    plt.savefig(fig_out, dpi=300)
    plt.close()
    print(f"Saved SHAP plot to {fig_out}")


# ============================================================
# 5. DOMAIN CONTRIBUTION ANALYSIS ACROSS TARGETS
# ============================================================

domain_df = pd.DataFrame(domain_records)
domain_summary = (
    domain_df.groupby(["Target", "Domain"])["Importance"]
    .sum()
    .reset_index()
)

# Normalize within each target to obtain percentage contribution per domain
target_totals = domain_summary.groupby("Target")["Importance"].transform("sum")
domain_summary["Percentage_Contribution"] = (domain_summary["Importance"] / target_totals) * 100

domain_csv_out = os.path.join(OUTPUT_DIR, "domain_importance_summary.csv")
domain_summary.to_csv(domain_csv_out, index=False)
print(f"\nSaved domain summary to {domain_csv_out}")

# Plot Domain Comparison
plt.figure(figsize=(12, 6))
chart = sns.barplot(
    data=domain_summary,
    x="Domain",
    y="Percentage_Contribution",
    hue="Target",
    palette=["#3498db", "#e74c3c", "#2ecc71"],
    edgecolor="#333333"
)
plt.title("Domain-Level Contribution to Student Mental Health Prediction (SHAP Share)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Psychosocial & Environmental Domain", fontsize=11, fontweight="bold")
plt.ylabel("Relative Domain Contribution (%)", fontsize=11, fontweight="bold")
plt.xticks(rotation=20, ha="right", fontsize=10)
plt.legend(title="Outcome", frameon=True, framealpha=0.9)
plt.tight_layout()

domain_fig_out = os.path.join(OUTPUT_DIR, "domain_importance_comparison.png")
plt.savefig(domain_fig_out, dpi=300)
plt.close()
print(f"Saved domain comparison figure to {domain_fig_out}")

print("\n" + "=" * 75)
print("STAGE 14 COMPLETE: SHAP INTERPRETABILITY & DOMAIN SUMMARY")
print("=" * 75)
print(domain_summary.pivot(index="Domain", columns="Target", values="Percentage_Contribution").round(2).to_string())
