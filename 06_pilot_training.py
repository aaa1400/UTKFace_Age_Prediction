
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import TerminateOnNaN


# =====================================================
# 1. Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_PATH = PROJECT_DIR / "models" / "age_prediction_initial.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

TRAIN_STEPS = 200
VALIDATION_STEPS = 50

print("=" * 60)
print("UTKFace - Pilot Training")
print("=" * 60)


# =====================================================
# 2. Load CSV Files
# =====================================================

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")

print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))


# =====================================================
# 3. Load Saved Model
# =====================================================

print("\nLoading initial model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =====================================================
# 4. Image Preprocessing
# =====================================================

def load_and_preprocess_image(image_path, age):

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

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, ages)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=1024,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
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

print("\nTraining configuration:")
print("Epochs:", 1)
print("Training steps:", TRAIN_STEPS)
print("Validation steps:", VALIDATION_STEPS)
print("Batch size:", BATCH_SIZE)

callbacks = [
    TerminateOnNaN()
]


# =====================================================
# 7. Pilot Training
# =====================================================

print("\nStarting pilot training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1,
    steps_per_epoch=TRAIN_STEPS,
    validation_steps=VALIDATION_STEPS,
    shuffle=False,
    callbacks=callbacks
)
# =====================================================
# 8. Save Pilot Model
# =====================================================

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PILOT_MODEL_PATH = MODEL_DIR / "age_prediction_pilot.keras"

model.save(PILOT_MODEL_PATH)

print("\nPilot model saved successfully!")
print("Path:", PILOT_MODEL_PATH)


# =====================================================
# 9. Training Results
# =====================================================

print("\n" + "=" * 60)
print("PILOT TRAINING RESULTS")
print("=" * 60)

for metric_name, values in history.history.items():

    print(
        f"{metric_name}: {values[-1]:.4f}"
    )

print("\nPilot training completed successfully!")