
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path


# =====================================================
# 1. Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_age_model.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

print("=" * 60)
print("UTKFace - Model Evaluation")
print("=" * 60)


# =====================================================
# 2. Load Test Dataset
# =====================================================

test_df = pd.read_csv(
    SPLIT_DIR / "test.csv"
)

print("\nTest images:", len(test_df))


# =====================================================
# 3. Load Best Model
# =====================================================

print("\nLoading best model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully!")


# =====================================================
# 4. Image Preprocessing
# =====================================================

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


# =====================================================
# 5. Create Test Dataset
# =====================================================

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


# =====================================================
# 6. Make Predictions
# =====================================================

print("\nPredicting ages...")

predictions = model.predict(
    test_dataset,
    verbose=1
).flatten()

actual_ages = test_df["age"].values.astype(
    "float32"
)


# =====================================================
# 7. Calculate Evaluation Metrics
# =====================================================

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


print("\n" + "=" * 60)
print("TEST SET EVALUATION RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# =====================================================
# 8. Save Predictions
# =====================================================

results_df = pd.DataFrame({

    "image_path": test_df["image_path"],

    "age_group": test_df["age_group"],

    "actual_age": actual_ages,

    "predicted_age": predictions,

    "absolute_error": absolute_errors

})

prediction_path = (
    MODEL_DIR / "test_predictions.csv"
)

results_df.to_csv(
    prediction_path,
    index=False
)

print("\nTest predictions saved:")
print(prediction_path)


# =====================================================
# 9. Evaluate Age Groups
# =====================================================

print("\n" + "=" * 60)
print("AGE GROUP EVALUATION")
print("=" * 60)

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

print(
    group_results_df.to_string(
        index=False
    )
)

group_results_path = (
    MODEL_DIR / "age_group_evaluation.csv"
)

group_results_df.to_csv(
    group_results_path,
    index=False
)

print("\nAge group results saved:")
print(group_results_path)


# =====================================================
# 10. Final Summary
# =====================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print("Total test images:", len(actual_ages))

print(
    "Actual age range:",
    f"{actual_ages.min():.0f} - {actual_ages.max():.0f}"
)

print(
    "Predicted age range:",
    f"{predictions.min():.2f} - {predictions.max():.2f}"
)

print("\nEvaluation completed successfully!")