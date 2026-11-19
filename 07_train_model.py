
import pandas as pd
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN
)


# =====================================================
# 1. Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL_PATH = (
    MODEL_DIR / "age_prediction_initial.keras"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "best_age_model.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

EPOCHS = 15

print("=" * 60)
print("UTKFace - Full Model Training")
print("=" * 60)



# =====================================================
# 2. Load CSV Files
# =====================================================

train_df = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

val_df = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)

print("\nDataset Information:")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# =====================================================
# 2.1 Calculate Age-Based Sample Weights
# =====================================================

print("\nCalculating age-based sample weights...")

group_counts = train_df["age_group"].value_counts()

if train_df["age_group"].isna().any():
    raise ValueError(
        "Some training images have missing age groups!"
    )

if train_df["age"].isna().any():
    raise ValueError(
        "Some training images have missing ages!"
    )


number_of_groups = len(group_counts)
total_samples = len(train_df)

# Inverse square-root frequency weighting
group_weights = (
    total_samples / (number_of_groups * group_counts)
) ** 0.5

# Normalize using the actual sample distribution
mean_weight = (
    group_counts * group_weights
).sum() / total_samples

group_weights = group_weights / mean_weight

# Limit extreme weights
group_weights = group_weights.clip(
    lower=0.5,
    upper=2.0
)

# Normalize again
mean_weight = (
    group_counts * group_weights
).sum() / total_samples

group_weights = group_weights / mean_weight

# Assign weights to training samples
train_df["sample_weight"] = (
    train_df["age_group"].map(group_weights)
)

print("\nAge group weights:")

for group, weight in group_weights.items():
    print(f"{group}: {weight:.3f}")

print("\nSample weights calculated successfully!")
print("Minimum weight:", train_df["sample_weight"].min())
print("Maximum weight:", train_df["sample_weight"].max())

# =====================================================
# 3. Load Initial Model
# =====================================================

print("\nLoading initial model...")

if not INITIAL_MODEL_PATH.is_file():

    raise FileNotFoundError(
        f"Initial model not found: {INITIAL_MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    INITIAL_MODEL_PATH
)

print("Initial model loaded successfully!")



# =====================================================
# 4. Image Preprocessing
# =====================================================

def load_and_preprocess_training_image(
    image_path,
    age,
    sample_weight
):

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

    age = tf.reshape(
        tf.cast(age, tf.float32),
        shape=(1,)
    )

    sample_weight = tf.cast(
        sample_weight,
        tf.float32
    )

    return image, age, sample_weight


def load_and_preprocess_validation_image(
    image_path,
    age
):

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

    age = tf.reshape(
        tf.cast(age, tf.float32),
        shape=(1,)
    )

    return image, age


# =====================================================
# 5. Create Dataset
# =====================================================

def create_dataset(df, training=False):

    image_paths = df["image_path"].values
    ages = df["age"].values.astype("float32")

    if training:

        sample_weights = df[
            "sample_weight"
        ].values.astype("float32")

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, ages, sample_weights)
        )

        dataset = dataset.shuffle(
            buffer_size=1024,
            reshuffle_each_iteration=True
        )

        dataset = dataset.map(
            load_and_preprocess_training_image,
            num_parallel_calls=1
        )

    else:

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, ages)
        )

        dataset = dataset.map(
            load_and_preprocess_validation_image,
            num_parallel_calls=1
        )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    dataset = dataset.prefetch(1)

    return dataset


train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

print("\nDatasets created successfully!")

# =====================================================
# 6. Training Configuration
# =====================================================

print("\nTraining Configuration:")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Training batches:", (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE)
print("Validation batches:", (len(val_df) + BATCH_SIZE - 1) // BATCH_SIZE)

callbacks = [

    ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    TerminateOnNaN()
]


# =====================================================
# 7. Full Model Training
# =====================================================

print("\nStarting full model training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)


# =====================================================
# 8. Save Final Training History
# =====================================================

history_df = pd.DataFrame(
    history.history
)

history_path = MODEL_DIR / "training_history.csv"

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved successfully!")
print("Path:", history_path)


# =====================================================
# 9. Training Results
# =====================================================

print("\n" + "=" * 60)
print("FULL TRAINING RESULTS")
print("=" * 60)

for metric_name, values in history.history.items():

    print(
        f"{metric_name}: {values[-1]:.4f}"
    )

print("\nBest model path:")
print(BEST_MODEL_PATH)

print("\nFull training completed successfully!")
