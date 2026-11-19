
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# =====================================================
# 1. Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = (160, 160, 3)

WEIGHTS_PATH = (
    PROJECT_DIR
    / "weights"
    / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
)

print("=" * 60)
print("UTKFace - MobileNetV2 Model Building")
print("=" * 60)

print("TensorFlow version:", tf.__version__)


# =====================================================
# 2. Check Local Weights File
# =====================================================

print("\nChecking pretrained weights file...")

if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(
        f"Pretrained weights not found: {WEIGHTS_PATH}"
    )

file_size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)

print("Weights file found!")
print("File size:", round(file_size_mb, 2), "MB")


# =====================================================
# 3. Build MobileNetV2 Architecture
# =====================================================

print("\nBuilding MobileNetV2 architecture...")

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE,
    include_top=False,
    weights=None
)

print("Architecture created successfully!")


# =====================================================
# 4. Load Local Pretrained Weights
# =====================================================

print("\nLoading local pretrained weights...")

base_model.load_weights(str(WEIGHTS_PATH))

print("Pretrained weights loaded successfully!")


# =====================================================
# 5. Freeze Base Model
# =====================================================

base_model.trainable = False

print("\nBase model frozen.")
print("Trainable:", base_model.trainable)


# =====================================================
# 6. Build Regression Model
# =====================================================

inputs = layers.Input(shape=IMAGE_SIZE)

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

outputs = layers.Dense(
    1,
    activation="linear",
    name="predicted_age"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="UTKFace_MobileNetV2_Age_Prediction"
)


# =====================================================
# 7. Compile Model
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

print("\nModel compiled successfully!")


# =====================================================
# 8. Model Summary
# =====================================================

print("\nModel Architecture:")
model.summary()


# =====================================================
# 9. Parameter Verification
# =====================================================

trainable_params = sum(
    tf.keras.backend.count_params(w)
    for w in model.trainable_weights
)

non_trainable_params = sum(
    tf.keras.backend.count_params(w)
    for w in model.non_trainable_weights
)

print("\n" + "=" * 60)
print("MODEL PARAMETER VERIFICATION")
print("=" * 60)

print("Total parameters:", f"{model.count_params():,}")
print("Trainable parameters:", f"{trainable_params:,}")
print("Non-trainable parameters:", f"{non_trainable_params:,}")

print("\nBase model trainable:", base_model.trainable)

print("\nModel building completed successfully!")
# =====================================================
# 10. Save Initial Model
# =====================================================

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "age_prediction_initial.keras"

model.save(MODEL_PATH)

print("\nInitial model saved successfully!")
print("Model path:", MODEL_PATH)
print("Model size:", round(
    MODEL_PATH.stat().st_size / (1024 * 1024), 2
), "MB")