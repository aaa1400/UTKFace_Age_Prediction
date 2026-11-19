
# ============================================================
# 12_evaluate_finetuned_model.py
# Evaluate and Compare Baseline and Fine-Tuned Models
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"

BASELINE_MODEL_PATH = MODEL_DIR / "best_age_model.keras"

FINETUNED_MODEL_PATH = (
    MODEL_DIR / "best_age_model_finetuned.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

BASELINE_PREDICTIONS_PATH = (
    MODEL_DIR / "baseline_test_predictions.csv"
)

FINETUNED_PREDICTIONS_PATH = (
    MODEL_DIR / "finetuned_test_predictions.csv"
)

COMPARISON_PATH = (
    MODEL_DIR / "model_comparison.csv"
)

FINETUNED_GROUP_RESULTS_PATH = (
    MODEL_DIR / "finetuned_age_group_evaluation.csv"
)

print("=" * 65)
print("UTKFace - Baseline vs Fine-Tuned Model Evaluation")
print("=" * 65)


# ------------------------------------------------------------
# 2. Check Required Files
# ------------------------------------------------------------

TEST_CSV = SPLIT_DIR / "test.csv"

for path in [
    TEST_CSV,
    BASELINE_MODEL_PATH,
    FINETUNED_MODEL_PATH
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("\nAll required files found!")


# ------------------------------------------------------------
# 3. Load Test Dataset
# ------------------------------------------------------------

test_df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(test_df))

required_columns = [
    "image_path",
    "age",
    "age_group"
]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

actual_ages = test_df["age"].values.astype(
    np.float32
)


# ------------------------------------------------------------
# 4. Image Preprocessing
# ------------------------------------------------------------

def load_and_preprocess_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image


# ------------------------------------------------------------
# 5. Create Test Dataset
# ------------------------------------------------------------

image_paths = test_df["image_path"].values

test_dataset = tf.data.Dataset.from_tensor_slices(
    image_paths
)

test_dataset = test_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=1
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(1)

print("\nTest dataset created successfully!")


# ------------------------------------------------------------
# 6. Evaluation Function
# ------------------------------------------------------------

def evaluate_model(model_path, model_name):

    print("\n" + "=" * 65)
    print(f"Evaluating: {model_name}")
    print("=" * 65)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        model_path
    )

    print("Model loaded successfully!")

    print("\nPredicting ages...")

    predictions = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    if len(predictions) != len(actual_ages):
        raise ValueError(
            "Prediction count does not match test samples!"
        )

    errors = actual_ages - predictions

    absolute_errors = np.abs(errors)

    mae = np.mean(absolute_errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(errors ** 2)

    ss_tot = np.sum(
        (actual_ages - np.mean(actual_ages)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    print("\nEvaluation Results")
    print("-" * 40)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    results_df = pd.DataFrame({

        "image_path": test_df["image_path"].values,

        "age_group": test_df["age_group"].values,

        "actual_age": actual_ages,

        "predicted_age": predictions,

        "absolute_error": absolute_errors

    })

    group_results = []

    for group, data in results_df.groupby("age_group"):

        group_mae = data["absolute_error"].mean()

        group_rmse = np.sqrt(
            np.mean(
                (
                    data["actual_age"]
                    - data["predicted_age"]
                ) ** 2
            )
        )

        group_results.append({

            "Age Group": group,

            "Samples": len(data),

            "MAE": group_mae,

            "RMSE": group_rmse

        })

    group_results_df = pd.DataFrame(
        group_results
    )

    del model

    return mae, rmse, r2, results_df, group_results_df


# ------------------------------------------------------------
# 7. Evaluate Baseline Model
# ------------------------------------------------------------

(
    baseline_mae,
    baseline_rmse,
    baseline_r2,
    baseline_results,
    baseline_groups
) = evaluate_model(
    BASELINE_MODEL_PATH,
    "Baseline Model"
)

baseline_results.to_csv(
    BASELINE_PREDICTIONS_PATH,
    index=False
)

print("\nBaseline predictions saved:")
print(BASELINE_PREDICTIONS_PATH)


# ------------------------------------------------------------
# 8. Evaluate Fine-Tuned Model
# ------------------------------------------------------------

(
    finetuned_mae,
    finetuned_rmse,
    finetuned_r2,
    finetuned_results,
    finetuned_groups
) = evaluate_model(
    FINETUNED_MODEL_PATH,
    "Fine-Tuned Model"
)

finetuned_results.to_csv(
    FINETUNED_PREDICTIONS_PATH,
    index=False
)

finetuned_groups.to_csv(
    FINETUNED_GROUP_RESULTS_PATH,
    index=False
)

print("\nFine-Tuned predictions saved:")
print(FINETUNED_PREDICTIONS_PATH)


# ------------------------------------------------------------
# 9. Compare Overall Performance
# ------------------------------------------------------------

comparison_df = pd.DataFrame({

    "Model": [
        "Baseline",
        "Fine-Tuned"
    ],

    "MAE": [
        baseline_mae,
        finetuned_mae
    ],

    "RMSE": [
        baseline_rmse,
        finetuned_rmse
    ],

    "R2": [
        baseline_r2,
        finetuned_r2
    ]

})

print("\n" + "=" * 65)
print("OVERALL MODEL COMPARISON")
print("=" * 65)

print(
    comparison_df.to_string(
        index=False
    )
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nComparison saved:")
print(COMPARISON_PATH)


# ------------------------------------------------------------
# 10. Compare Age Groups
# ------------------------------------------------------------

age_comparison = baseline_groups[
    ["Age Group", "Samples", "MAE", "RMSE"]
].merge(
    finetuned_groups[
        ["Age Group", "MAE", "RMSE"]
    ],
    on="Age Group",
    suffixes=("_Baseline", "_FineTuned")
)

age_comparison["MAE_Improvement"] = (
    age_comparison["MAE_Baseline"]
    - age_comparison["MAE_FineTuned"]
)

age_comparison["MAE_Improvement_Percent"] = (
    age_comparison["MAE_Improvement"]
    / age_comparison["MAE_Baseline"]
) * 100

print("\n" + "=" * 65)
print("AGE GROUP PERFORMANCE COMPARISON")
print("=" * 65)

print(
    age_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 11. Final Summary
# ------------------------------------------------------------

mae_improvement = baseline_mae - finetuned_mae

improvement_percent = (
    mae_improvement / baseline_mae
) * 100

print("\n" + "=" * 65)
print("FINAL EVALUATION SUMMARY")
print("=" * 65)

print(f"Baseline Test MAE:   {baseline_mae:.4f}")
print(f"Fine-Tuned Test MAE: {finetuned_mae:.4f}")

print(f"MAE Improvement:    {mae_improvement:.4f}")
print(f"Improvement (%):    {improvement_percent:.2f}%")

if mae_improvement > 0:
    print("\nFine-Tuning improved test MAE!")
elif mae_improvement < 0:
    print("\nFine-Tuning did not improve test MAE.")
else:
    print("\nBoth models have equal test MAE.")

print("\nAge group comparison completed!")

print("\n" + "=" * 65)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)