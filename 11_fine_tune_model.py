
# ============================================================
# 11_fine_tune_model.py
# Fine-Tuning MobileNetV2 for UTKFace Age Prediction
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
TRAIN_CSV = PROJECT_DIR / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "splits" / "validation.csv"

BASE_MODEL_PATH = MODEL_DIR / "best_age_model.keras"
FINE_TUNED_MODEL_PATH = MODEL_DIR / "best_age_model_finetuned.keras"
HISTORY_PATH = MODEL_DIR / "fine_tuning_history.csv"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
UNFREEZE_LAYERS = 20


# ------------------------------------------------------------
# 2. Check files
# ------------------------------------------------------------

print("=" * 60)
print("UTKFace MobileNetV2 Fine-Tuning")
print("=" * 60)

for path in [BASE_MODEL_PATH, TRAIN_CSV, VAL_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("\nRequired files found!")


# ------------------------------------------------------------
# 3. Load datasets
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))


# ------------------------------------------------------------
# 4. Recalculate sample weights
# ------------------------------------------------------------

group_counts = train_df["age_group"].value_counts()

number_of_groups = len(group_counts)
total_samples = len(train_df)

group_weights = (
    total_samples / (number_of_groups * group_counts)
) ** 0.5

mean_weight = (
    group_counts * group_weights
).sum() / total_samples

group_weights = group_weights / mean_weight
group_weights = group_weights.clip(lower=0.5, upper=2.0)

mean_weight = (
    group_counts * group_weights
).sum() / total_samples

group_weights = group_weights / mean_weight

train_df["sample_weight"] = train_df["age_group"].map(group_weights)

if train_df["sample_weight"].isna().any():
    raise ValueError("Some training samples have missing weights!")

print("\nSample weights calculated successfully!")


# ------------------------------------------------------------
# 5. Build TensorFlow datasets
# ------------------------------------------------------------

def load_train_image(image_path, age, sample_weight):

    image_data = tf.io.read_file(image_path)

    image_data = tf.io.decode_jpeg(
        image_data,
        channels=3
    )

    image_data = tf.image.resize(
        image_data,
        IMAGE_SIZE
    )

    image_data = preprocess_input(image_data)

    age = tf.cast(age, tf.float32)
    age = tf.reshape(age, (1,))

    sample_weight = tf.cast(sample_weight, tf.float32)

    return image_data, age, sample_weight


def load_val_image(image_path, age):

    image_data = tf.io.read_file(image_path)

    image_data = tf.io.decode_jpeg(
        image_data,
        channels=3
    )

    image_data = tf.image.resize(
        image_data,
        IMAGE_SIZE
    )

    image_data = preprocess_input(image_data)

    age = tf.cast(age, tf.float32)
    age = tf.reshape(age, (1,))

    return image_data, age


def create_train_dataset(dataframe):

    paths = dataframe["image_path"].values
    ages = dataframe["age"].values.astype(np.float32)
    weights = dataframe["sample_weight"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, ages, weights)
    )

    dataset = dataset.shuffle(
        buffer_size=1024,
        seed=42
    )

    dataset = dataset.map(
        load_train_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


def create_val_dataset(dataframe):

    paths = dataframe["image_path"].values
    ages = dataframe["age"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, ages)
    )

    dataset = dataset.map(
        load_val_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


train_dataset = create_train_dataset(train_df)
val_dataset = create_val_dataset(val_df)

print("\nTensorFlow datasets created successfully!")


# ------------------------------------------------------------
# 6. Load best baseline model
# ------------------------------------------------------------

print("\nLoading baseline model...")

model = tf.keras.models.load_model(BASE_MODEL_PATH)

print("Baseline model loaded successfully!")


# ------------------------------------------------------------
# 7. Find MobileNetV2 backbone
# ------------------------------------------------------------

# base_model = None
print("\nConfiguring MobileNetV2 for Fine-Tuning...")

base_model = model.get_layer("mobilenetv2_1.00_160")

# Unfreeze the backbone
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep all BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Count trainable layers
trainable_layers = [
    layer for layer in base_model.layers
    if layer.trainable
]

print("Total Backbone Layers:", len(base_model.layers))
print("Trainable Backbone Layers:", len(trainable_layers))
print("BatchNormalization layers remain frozen.")

print("\nTrainable Layers:")
for layer in trainable_layers:
    print(layer.name, "|", type(layer).__name__)

print("\nFine-Tuning preparation completed!")
# for layer in model.layers:
#     if isinstance(layer, tf.keras.Model):
#         if "mobilenetv2" in layer.name.lower():
#             base_model = layer
#             break

# if base_model is None:
#     raise ValueError("MobileNetV2 backbone was not found!")

# print("\nBackbone found:", base_model.name)
# print("Total backbone layers:", len(base_model.layers))


# # ------------------------------------------------------------
# # 8. Unfreeze last 20 layers
# # ------------------------------------------------------------

# base_model.trainable = True

# for layer in base_model.layers[:-UNFREEZE_LAYERS]:
#     layer.trainable = False

# for layer in base_model.layers:
#     if isinstance(layer, tf.keras.layers.BatchNormalization):
#         layer.trainable = False

# trainable_layers = [
#     layer for layer in base_model.layers
#     if layer.trainable
# ]

# print("\nFine-Tuning Configuration")
# print("-" * 40)
# print("Total backbone layers:", len(base_model.layers))
# print("Trainable backbone layers:", len(trainable_layers))
# print("BatchNormalization layers remain frozen.")


# ------------------------------------------------------------
# 9. Recompile model with a low learning rate
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

print("\nModel compiled successfully!")
print("Learning Rate:", LEARNING_RATE)


# ------------------------------------------------------------
# 10. Callbacks
# ------------------------------------------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=FINE_TUNED_MODEL_PATH,
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.TerminateOnNaN()
]


# ------------------------------------------------------------
# 11. Start Fine-Tuning
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    shuffle=False
)


# ------------------------------------------------------------
# 12. Save training history
# ------------------------------------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print("\nFine-Tuning history saved:", HISTORY_PATH)


# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETED")
print("=" * 60)

if "val_mae" in history_df.columns:

    best_epoch = history_df["val_mae"].idxmin() + 1
    best_val_mae = history_df["val_mae"].min()

    print("Best Epoch:", best_epoch)
    print(f"Best Validation MAE: {best_val_mae:.4f}")

print("\nFine-Tuned model path:")
print(FINE_TUNED_MODEL_PATH)

print("\nAll tasks completed!")