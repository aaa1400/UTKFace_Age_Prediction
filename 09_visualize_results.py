
# ============================================================
# 09_visualize_results.py
# Visualize UTKFace Age Prediction Model Results
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

HISTORY_PATH = PROJECT_DIR / "models" / "training_history.csv"
PREDICTIONS_PATH = PROJECT_DIR / "models" / "test_predictions.csv"
AGE_GROUP_PATH = PROJECT_DIR / "models" / "age_group_evaluation.csv"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load CSV files
# ------------------------------------------------------------

print("=" * 60)
print("UTKFace Model Results Visualization")
print("=" * 60)

for file_path in [HISTORY_PATH, PREDICTIONS_PATH, AGE_GROUP_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

history_df = pd.read_csv(HISTORY_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)
age_group_df = pd.read_csv(AGE_GROUP_PATH)

print("\nCSV files loaded successfully!")

print("Training history shape:", history_df.shape)
print("Test predictions shape:", predictions_df.shape)
print("Age group evaluation shape:", age_group_df.shape)


# ------------------------------------------------------------
# 3. Verify required columns
# ------------------------------------------------------------

required_history = ["loss", "val_loss", "mae", "val_mae"]
required_predictions = ["actual_age", "predicted_age"]
required_age_group = ["Age Group", "MAE"]

for column in required_history:
    if column not in history_df.columns:
        raise ValueError(f"Missing column in training history: {column}")

for column in required_predictions:
    if column not in predictions_df.columns:
        raise ValueError(f"Missing column in predictions: {column}")

for column in required_age_group:
    if column not in age_group_df.columns:
        raise ValueError(f"Missing column in age group evaluation: {column}")

print("\nRequired columns verified successfully!")


# ------------------------------------------------------------
# 4. Plot Training Loss vs Validation Loss
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

epochs = range(1, len(history_df) + 1)

plt.plot(
    epochs,
    history_df["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history_df["val_loss"],
    marker="s",
    label="Validation Loss"
)

plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "training_validation_loss.png",
    dpi=300
)

plt.close()

print("\n1. Loss comparison plot saved.")


# ------------------------------------------------------------
# 5. Plot Training MAE vs Validation MAE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history_df["mae"],
    marker="o",
    label="Training MAE"
)

plt.plot(
    epochs,
    history_df["val_mae"],
    marker="s",
    label="Validation MAE"
)

plt.title("Training MAE vs Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (Years)")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "training_validation_mae.png",
    dpi=300
)

plt.close()

print("2. MAE comparison plot saved.")


# ------------------------------------------------------------
# 6. Plot Actual Age vs Predicted Age
# ------------------------------------------------------------

actual_ages = predictions_df["actual_age"]
predicted_ages = predictions_df["predicted_age"]

plt.figure(figsize=(9, 9))

plt.scatter(
    actual_ages,
    predicted_ages,
    alpha=0.4
)

min_age = min(actual_ages.min(), predicted_ages.min())
max_age = max(actual_ages.max(), predicted_ages.max())

plt.plot(
    [min_age, max_age],
    [min_age, max_age],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual Age vs Predicted Age")
plt.xlabel("Actual Age (Years)")
plt.ylabel("Predicted Age (Years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "actual_vs_predicted_age.png",
    dpi=300
)

plt.close()

print("3. Actual vs Predicted plot saved.")


# ------------------------------------------------------------
# 7. Plot MAE for Different Age Groups
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.bar(
    age_group_df["Age Group"].astype(str),
    age_group_df["MAE"]
)

plt.title("Model MAE Across Different Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Mean Absolute Error (Years)")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "mae_per_age_group.png",
    dpi=300
)

plt.close()

print("4. Age group MAE plot saved.")


# ------------------------------------------------------------
# 8. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

for image_path in sorted(REPORT_DIR.glob("*.png")):
    print(f"[OK] {image_path.name}")

print("\nReport directory:")
print(REPORT_DIR)

print("\nAll visualization tasks completed!")