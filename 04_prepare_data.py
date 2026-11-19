
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "splits"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SHUFFLE_BUFFER = 1024

print("=" * 55)
print("UTKFace - Data Preparation")
print("=" * 55)

print("TensorFlow version:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)


# =====================================================
# 2. Load CSV files
# =====================================================

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =====================================================
# 3. Check image paths
# =====================================================

for name, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    missing = sum(
        not (PROJECT_DIR / path).is_file()
        for path in df["image_path"]
    )

    print(f"{name} - Missing images: {missing}")

    if missing > 0:
        raise FileNotFoundError(
            f"{name} dataset contains missing images."
        )


# =====================================================
# 4. Image loading and preprocessing
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

    image = preprocess_input(image)

    # Shape: (1,) to match model output (batch, 1)
    age = tf.reshape(
        tf.cast(age, tf.float32),
        shape=(1,)
    )

    return image, age


# =====================================================
# 5. Create TensorFlow Dataset
# =====================================================

def create_dataset(df, training=False):

    image_paths = df["image_path"].values
    ages = df["age"].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, ages)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER,
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


# =====================================================
# 6. Build Train, Validation and Test datasets
# =====================================================

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)

print("\nTensorFlow datasets created successfully!")


# =====================================================
# 7. Inspect one batch
# =====================================================

print("\n" + "=" * 55)
print("BATCH VERIFICATION")
print("=" * 55)

images, ages = next(iter(train_dataset))

print("\nTrain batch:")
print("Images shape:", images.shape)
print("Images dtype:", images.dtype)

print("Ages shape:", ages.shape)
print("Ages dtype:", ages.dtype)

print("Image minimum:", float(tf.reduce_min(images)))
print("Image maximum:", float(tf.reduce_max(images)))

print("First 5 ages:")
print(ages[:5].numpy().flatten())

print("\nExpected image shape:")
print("(8, 160, 160, 3)")

print("\nExpected age shape:")
print("(8, 1)")


# =====================================================
# 8. Verify Validation and Test datasets
# =====================================================

val_images, val_ages = next(iter(val_dataset))
test_images, test_ages = next(iter(test_dataset))

print("\nValidation batch:")
print("Images:", val_images.shape)
print("Ages:", val_ages.shape)

print("\nTest batch:")
print("Images:", test_images.shape)
print("Ages:", test_ages.shape)


# =====================================================
# 9. Final status
# =====================================================

print("\n" + "=" * 55)
print("DATA PREPARATION COMPLETED!")
print("=" * 55)

print("Train batches:", len(train_dataset))
print("Validation batches:", len(val_dataset))
print("Test batches:", len(test_dataset))