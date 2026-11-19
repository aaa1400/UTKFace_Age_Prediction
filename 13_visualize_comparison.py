
# ============================================================
# 13_visualize_comparison.py
# Baseline vs Fine-Tuned Model Visualization
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

BASELINE_PREDICTIONS_PATH = (
    MODEL_DIR / "baseline_test_predictions.csv"
)

FINETUNED_PREDICTIONS_PATH = (
    MODEL_DIR / "finetuned_test_predictions.csv"
)

BASELINE_GROUP_PATH = (
    MODEL_DIR / "age_group_evaluation.csv"
)

FINETUNED_GROUP_PATH = (
    MODEL_DIR / "finetuned_age_group_evaluation.csv"
)

AGE_GROUP_ORDER = [
    "0-3",
    "4-11",
    "12-17",
    "18-29",
    "30-44",
    "45-59",
    "60-74",
    "75+"
]

DPI = 300

print("=" * 65)
print("UTKFace - Model Comparison Visualization")
print("=" * 65)


# ------------------------------------------------------------
# 2. Check Required Files
# ------------------------------------------------------------

required_files = [
    COMPARISON_PATH,
    BASELINE_PREDICTIONS_PATH,
    FINETUNED_PREDICTIONS_PATH,
    BASELINE_GROUP_PATH,
    FINETUNED_GROUP_PATH
]

for file_path in required_files:

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\nAll required files found!")


# ------------------------------------------------------------
# 3. Load Evaluation Results
# ------------------------------------------------------------

comparison_df = pd.read_csv(COMPARISON_PATH)

baseline_df = pd.read_csv(
    BASELINE_PREDICTIONS_PATH
)

finetuned_df = pd.read_csv(
    FINETUNED_PREDICTIONS_PATH
)

baseline_groups = pd.read_csv(
    BASELINE_GROUP_PATH
)

finetuned_groups = pd.read_csv(
    FINETUNED_GROUP_PATH
)

print("\nCSV files loaded successfully!")

print("Model comparison shape:", comparison_df.shape)
print("Baseline predictions shape:", baseline_df.shape)
print("Fine-Tuned predictions shape:", finetuned_df.shape)


# ------------------------------------------------------------
# 4. Validate Required Columns
# ------------------------------------------------------------

required_comparison_columns = [
    "Model",
    "MAE",
    "RMSE",
    "R2"
]

required_prediction_columns = [
    "actual_age",
    "predicted_age",
    "absolute_error"
]

required_group_columns = [
    "Age Group",
    "Samples",
    "MAE",
    "RMSE"
]

for column in required_comparison_columns:

    if column not in comparison_df.columns:
        raise ValueError(
            f"Missing comparison column: {column}"
        )

for dataframe_name, dataframe in [
    ("Baseline predictions", baseline_df),
    ("Fine-Tuned predictions", finetuned_df)
]:

    for column in required_prediction_columns:

        if column not in dataframe.columns:
            raise ValueError(
                f"{dataframe_name}: missing column {column}"
            )

for dataframe_name, dataframe in [
    ("Baseline age groups", baseline_groups),
    ("Fine-Tuned age groups", finetuned_groups)
]:

    for column in required_group_columns:

        if column not in dataframe.columns:
            raise ValueError(
                f"{dataframe_name}: missing column {column}"
            )

if len(baseline_df) != len(finetuned_df):
    raise ValueError(
        "Baseline and Fine-Tuned prediction counts differ!"
    )

if not np.allclose(
    baseline_df["actual_age"].values,
    finetuned_df["actual_age"].values
):
    raise ValueError(
        "Actual ages do not match between models!"
    )

print("\nRequired columns verified successfully!")
print("Prediction datasets are aligned!")


# ------------------------------------------------------------
# 5. Overall Model Metrics Comparison
# ------------------------------------------------------------

print("\nCreating overall metrics comparison plot...")

baseline_metrics = comparison_df[
    comparison_df["Model"] == "Baseline"
].iloc[0]

finetuned_metrics = comparison_df[
    comparison_df["Model"] == "Fine-Tuned"
].iloc[0]

metric_names = ["MAE", "RMSE", "R2"]

baseline_values = [
    baseline_metrics["MAE"],
    baseline_metrics["RMSE"],
    baseline_metrics["R2"]
]

finetuned_values = [
    finetuned_metrics["MAE"],
    finetuned_metrics["RMSE"],
    finetuned_metrics["R2"]
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

for index, metric in enumerate(metric_names):

    values = [
        baseline_values[index],
        finetuned_values[index]
    ]

    bars = axes[index].bar(
        ["Baseline", "Fine-Tuned"],
        values
    )

    axes[index].set_title(
        metric,
        fontsize=13,
        fontweight="bold"
    )

    axes[index].set_ylabel(metric)

    axes[index].grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    axes[index].set_axisbelow(True)

    for bar in bars:

        height = bar.get_height()

        axes[index].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

fig.suptitle(
    "Baseline vs Fine-Tuned Model Performance",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

metrics_plot_path = (
    REPORT_DIR / "model_metrics_comparison.png"
)

plt.savefig(
    metrics_plot_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print("1. Overall metrics comparison saved!")


# ------------------------------------------------------------
# 6. Age Group MAE Comparison
# ------------------------------------------------------------

print("\nCreating age group comparison plot...")

age_comparison = baseline_groups[
    ["Age Group", "MAE"]
].merge(
    finetuned_groups[
        ["Age Group", "MAE"]
    ],
    on="Age Group",
    suffixes=("_Baseline", "_FineTuned")
)

age_comparison["Age Group"] = pd.Categorical(
    age_comparison["Age Group"],
    categories=AGE_GROUP_ORDER,
    ordered=True
)

age_comparison = age_comparison.sort_values(
    "Age Group"
)

if age_comparison["MAE_Baseline"].isna().any():
    raise ValueError("Missing baseline age group results!")

if age_comparison["MAE_FineTuned"].isna().any():
    raise ValueError("Missing Fine-Tuned age group results!")

x = np.arange(len(age_comparison))

width = 0.36

fig, ax = plt.subplots(
    figsize=(12, 6)
)

bars1 = ax.bar(
    x - width / 2,
    age_comparison["MAE_Baseline"],
    width,
    label="Baseline"
)

bars2 = ax.bar(
    x + width / 2,
    age_comparison["MAE_FineTuned"],
    width,
    label="Fine-Tuned"
)

ax.set_title(
    "MAE Comparison Across Age Groups",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Age Group")

ax.set_ylabel("Mean Absolute Error (Years)")

ax.set_xticks(x)

ax.set_xticklabels(
    age_comparison["Age Group"].astype(str)
)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)

ax.bar_label(
    bars1,
    fmt="%.2f",
    padding=2,
    fontsize=8
)

ax.bar_label(
    bars2,
    fmt="%.2f",
    padding=2,
    fontsize=8
)

plt.tight_layout()

age_group_plot_path = (
    REPORT_DIR / "age_group_model_comparison.png"
)

plt.savefig(
    age_group_plot_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print("2. Age group comparison plot saved!")


# ------------------------------------------------------------
# 7. Actual vs Predicted - Fine-Tuned Model
# ------------------------------------------------------------

print("\nCreating Actual vs Predicted plot...")

actual_ages = finetuned_df["actual_age"].values

predicted_ages = finetuned_df["predicted_age"].values

mae = finetuned_metrics["MAE"]
rmse = finetuned_metrics["RMSE"]
r2 = finetuned_metrics["R2"]

min_age = min(
    actual_ages.min(),
    predicted_ages.min()
)

max_age = max(
    actual_ages.max(),
    predicted_ages.max()
)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.scatter(
    actual_ages,
    predicted_ages,
    alpha=0.35,
    s=14,
    label="Predictions"
)

ax.plot(
    [min_age, max_age],
    [min_age, max_age],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

ax.set_title(
    "Actual vs Predicted Age - Fine-Tuned Model",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Actual Age (Years)")

ax.set_ylabel("Predicted Age (Years)")

ax.legend()

ax.grid(
    linestyle="--",
    alpha=0.35
)

metrics_text = (
    f"MAE: {mae:.3f} years\n"
    f"RMSE: {rmse:.3f} years\n"
    f"R2: {r2:.3f}"
)

ax.text(
    0.05,
    0.95,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.85
    ),
    fontsize=11
)

plt.tight_layout()

actual_predicted_plot_path = (
    REPORT_DIR / "finetuned_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_plot_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print("3. Fine-Tuned Actual vs Predicted plot saved!")


# ------------------------------------------------------------
# 8. Final Summary
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated plots:")

for plot_path in [
    metrics_plot_path,
    age_group_plot_path,
    actual_predicted_plot_path
]:

    print("\n", plot_path)
    print("File exists:", plot_path.is_file())

    if plot_path.is_file():

        size_kb = plot_path.stat().st_size / 1024

        print(f"File size: {size_kb:.2f} KB")

print("\nAll visualizations saved successfully!")
