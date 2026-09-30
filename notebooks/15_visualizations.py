# ============================================================
# 15_visualizations.py
# Publication-Quality Figures for Research Paper
# - Figure 1: Multi-target Normalized Confusion Matrices (3x1 panel)
# - Figure 2: Model Progression & Benchmark Comparison
# - Figure 3: Per-Class Sensitivity / Recall Profiles across Ordinal Severity
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe", "Extremely"]
TARGETS = ["DASS_Stress", "DASS-Anxiety", "DASS-Depression"]
TARGET_TITLES = {
    "DASS_Stress": "DASS Stress (Logistic Regression)",
    "DASS-Anxiety": "DASS Anxiety (Logistic Regression)",
    "DASS-Depression": "DASS Depression (Random Forest)"
}


# ============================================================
# 1. FIGURE: NORMALIZED CONFUSION MATRICES PANEL
# ============================================================

cm_df = pd.read_csv(os.path.join(OUTPUT_DIR, "final_model_confusion_matrices.csv"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for i, target in enumerate(TARGETS):
    sub_cm = cm_df[cm_df["Target"] == target]
    matrix = np.zeros((5, 5))
    for _, row in sub_cm.iterrows():
        matrix[int(row["True_Class"]), int(row["Predicted_Class"])] = row["Count"]

    # Normalize by true class row sums
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    sns.heatmap(
        norm_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True if i == 2 else False,
        ax=axes[i],
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#dddddd"
    )

    axes[i].set_title(TARGET_TITLES[target], fontsize=12, fontweight="bold", pad=10)
    axes[i].set_xlabel("Predicted Severity Level", fontsize=11, fontweight="bold")
    if i == 0:
        axes[i].set_ylabel("True Severity Level", fontsize=11, fontweight="bold")
    else:
        axes[i].set_ylabel("")

plt.tight_layout()
cm_fig_path = os.path.join(OUTPUT_DIR, "final_confusion_matrices.png")
plt.savefig(cm_fig_path, dpi=300)
plt.close()
print(f"Saved: {cm_fig_path}")


# ============================================================
# 2. FIGURE: PER-CLASS RECALL PROFILES ACROSS ORDINAL LEVELS
# ============================================================

recall_df = pd.read_csv(os.path.join(OUTPUT_DIR, "final_model_class_recall.csv"))
recall_summary = (
    recall_df.groupby(["Target", "Class", "Class_Name"])["Recall"]
    .agg(mean="mean", std=lambda x: np.std(x, ddof=1))
    .reset_index()
)

plt.figure(figsize=(9, 5.5))
palette = {"DASS_Stress": "#3498db", "DASS-Anxiety": "#e74c3c", "DASS-Depression": "#2ecc71"}
markers = {"DASS_Stress": "o", "DASS-Anxiety": "s", "DASS-Depression": "^"}

for target in TARGETS:
    sub = recall_summary[recall_summary["Target"] == target]
    plt.errorbar(
        sub["Class"],
        sub["mean"],
        yerr=sub["std"],
        label=target.replace("-", " "),
        color=palette[target],
        marker=markers[target],
        markersize=7,
        capsize=4,
        linewidth=2,
        alpha=0.9
    )

plt.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="Chance Level (1/5)")
plt.xticks(ticks=range(5), labels=CLASS_NAMES, fontsize=10, fontweight="bold")
plt.xlabel("DASS Severity Level (Ordered)", fontsize=11, fontweight="bold")
plt.ylabel("Cross-Validated Recall (Mean ± STD)", fontsize=11, fontweight="bold")
plt.title("Per-Class Recall Across Ordinal Severity Spectra", fontsize=13, fontweight="bold", pad=12)
plt.ylim(-0.05, 1.05)
plt.legend(frameon=True, framealpha=0.95, loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

recall_fig_path = os.path.join(OUTPUT_DIR, "per_class_recall_comparison.png")
plt.savefig(recall_fig_path, dpi=300)
plt.close()
print(f"Saved: {recall_fig_path}")


# ============================================================
# 3. FIGURE: MODEL PROGRESSION BENCHMARK
# ============================================================

# Compare Dummy Baseline vs Final Selected Model
baseline_df = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_model_summary_corrected.csv"))
final_df = pd.read_csv(os.path.join(OUTPUT_DIR, "final_model_selection_summary.csv"))

dummy_sub = baseline_df[baseline_df["Model"] == "Dummy"][["Target", "Macro_F1_Mean", "Balanced_Accuracy_Mean", "Ordinal_MAE_Mean"]].copy()
dummy_sub["Stage"] = "Majority Dummy Baseline"

final_sub = final_df[["Target", "Macro_F1_Mean", "Balanced_Accuracy_Mean", "Ordinal_MAE_Mean"]].copy()
final_sub["Stage"] = "Final Optimized Model"

comparison_df = pd.concat([dummy_sub, final_sub], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ("Macro_F1_Mean", "Macro-F1 (Higher is Better)", axes[0]),
    ("Balanced_Accuracy_Mean", "Balanced Accuracy (Higher is Better)", axes[1]),
    ("Ordinal_MAE_Mean", "Ordinal MAE (Lower is Better)", axes[2])
]

for col, label, ax in metrics:
    sns.barplot(
        data=comparison_df,
        x="Target",
        y=col,
        hue="Stage",
        palette=["#95a5a6", "#2c3e50"],
        ax=ax,
        edgecolor="#333333"
    )
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    if ax != axes[0]:
        ax.get_legend().remove()
    else:
        ax.legend(title="", frameon=True, framealpha=0.9)

plt.tight_layout()
benchmark_fig_path = os.path.join(OUTPUT_DIR, "model_benchmark_progression.png")
plt.savefig(benchmark_fig_path, dpi=300)
plt.close()
print(f"Saved: {benchmark_fig_path}")

print("\nVisualizations complete!")
