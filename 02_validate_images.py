
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# ==========================================
# 1. Configuration
# ==========================================

DATASET_DIR = Path("dataset/UTKFace")

valid_images = []
invalid_images = []

image_modes = Counter()
image_sizes = Counter()
image_formats = Counter()

# ==========================================
# 2. Scan Images
# ==========================================

image_files = sorted(DATASET_DIR.glob("*.jpg"))

print("=" * 55)
print("UTKFace Image Validation")
print("=" * 55)

print("Total images:", len(image_files))

# ==========================================
# 3. Validate Each Image
# ==========================================

for index, image_path in enumerate(image_files, start=1):

    try:
        # First check image integrity
        with Image.open(image_path) as img:
            img.verify()

        # Reopen and fully load the image
        with Image.open(image_path) as img:
            img.load()

            width, height = img.size
            mode = img.mode
            image_format = img.format

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            valid_images.append({
                "filename": image_path.name,
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format
            })

            image_modes[mode] += 1
            image_sizes[(width, height)] += 1
            image_formats[image_format] += 1

    except Exception as error:

        invalid_images.append({
            "filename": image_path.name,
            "error": str(error)
        })

    # Progress report
    if index % 2000 == 0:
        print(f"Processed: {index}/{len(image_files)}")

# ==========================================
# 4. Validation Summary
# ==========================================

print("\n" + "=" * 55)
print("Validation Results")
print("=" * 55)

print("Total images:", len(image_files))
print("Valid images:", len(valid_images))
print("Invalid images:", len(invalid_images))

# ==========================================
# 5. Image Mode Analysis
# ==========================================

print("\nImage Modes")
print("-" * 55)

for mode, count in image_modes.items():
    print(f"{mode}: {count}")

# ==========================================
# 6. Image Format Analysis
# ==========================================

print("\nImage Formats")
print("-" * 55)

for image_format, count in image_formats.items():
    print(f"{image_format}: {count}")

# ==========================================
# 7. Image Size Analysis
# ==========================================

print("\nMost Common Image Sizes")
print("-" * 55)

for size, count in image_sizes.most_common(10):
    print(f"{size}: {count}")

# ==========================================
# 8. Save Reports
# ==========================================

pd.DataFrame(valid_images).to_csv(
    "valid_images_report.csv",
    index=False
)

pd.DataFrame(invalid_images).to_csv(
    "invalid_images_report.csv",
    index=False
)

print("\nReports saved successfully!")
print("\nImage validation completed!")