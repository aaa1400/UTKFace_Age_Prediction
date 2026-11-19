
# ============================================================
# 10_predict_age.py
# UTKFace Age Prediction - Single Image Inference
# ============================================================

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "best_age_model.keras"

IMAGE_SIZE = (160, 160)

# Change this path to your test image
IMAGE_PATH = PROJECT_DIR / "test_face.jpg"


# ------------------------------------------------------------
# 2. Check required files
# ------------------------------------------------------------

print("=" * 60)
print("UTKFace Age Prediction")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )

print("\nModel file found!")
print("Input image found!")


# ------------------------------------------------------------
# 3. Load trained model
# ------------------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ------------------------------------------------------------
# 4. Load and preprocess input image
# ------------------------------------------------------------

print("\nLoading input image...")

img = image.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Apply MobileNetV2 preprocessing
img_array = preprocess_input(img_array)

print("Image preprocessing completed!")

print("Input shape:", img_array.shape)
print("Input dtype:", img_array.dtype)


# ------------------------------------------------------------
# 5. Predict age
# ------------------------------------------------------------

print("\nPredicting age...")

prediction = model.predict(
    img_array,
    verbose=0
)

predicted_age = float(prediction[0][0])

# Keep the displayed age within the dataset age range
predicted_age = float(np.clip(predicted_age, 0, 116))

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"Predicted Age: {predicted_age:.2f} years")

print("=" * 60)


# ------------------------------------------------------------
# 6. Display input image and prediction
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.imshow(img)
plt.title(
    f"Predicted Age: {predicted_age:.1f} years"
)

plt.axis("off")
plt.tight_layout()

plt.show()

print("\nAge prediction completed successfully!")


# .........................................................
# (.venv) PS C:\AI_Projects\UTKFace_Age_Prediction> python 10_predict_age.py
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1795104095.193580    9644 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1795104104.267494    9644 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# ============================================================
# UTKFace Age Prediction
# ============================================================

# Model file found!
# Input image found!

# Loading trained model...
# I0000 00:00:1795104109.847123    9644 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.Model loaded successfully!

# Loading input image...
# Image preprocessing completed!
# Input shape: (1, 160, 160, 3)
# Input dtype: float32

# Predicting age...

# ============================================================
# PREDICTION RESULT
# ============================================================
# Predicted Age: 2.97 years
# ============================================================

# Age prediction completed successfully!