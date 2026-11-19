
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Dataset Configuration
# ==========================================

DATASET_DIR = Path("dataset/UTKFace")

# ==========================================
# 2. Read Image Filenames
# ==========================================

image_files = list(DATASET_DIR.glob("*.jpg"))

print("=" * 55)
print("UTKFace Dataset Exploration")
print("=" * 55)

print(f"\nTotal image files: {len(image_files)}")

# ==========================================
# 3. Extract Age Information
# ==========================================

records = []
invalid_files = []

for image_path in image_files:

    filename = image_path.name

    try:
        parts = filename.split("_")

        # Extract age and gender
        age = int(parts[0])
        gender = int(parts[1])

        # Validate age
        if not 0 <= age <= 116:
            raise ValueError("Invalid age")

        # Validate gender
        if gender not in [0, 1]:
            raise ValueError("Invalid gender")

        # Extract race if available
        if len(parts) >= 4:
            race = int(parts[2])
        else:
            race = None

        # Extract timestamp
        timestamp = parts[3] if len(parts) >= 4 else parts[2]

        records.append({
            "filename": filename,
            "age": age,
            "gender": gender,
            "race": race,
            "timestamp": timestamp
        })

    except (ValueError, IndexError):
        invalid_files.append(filename)

# ==========================================
# 4. Create DataFrame
# ==========================================

df = pd.DataFrame(records)

print("\nData Extraction Results")
print("-" * 55)

print("Valid images:", len(df))
print("Invalid images:", len(invalid_files))

# ==========================================
# 3.1 Display Invalid Filenames
# ==========================================

print("\nInvalid Filenames")
print("-" * 55)

for filename in invalid_files:
    print(filename)

# Save invalid filenames
with open("invalid_files.txt", "w", encoding="utf-8") as file:
    for filename in invalid_files:
        file.write(filename + "\n")

print("\nInvalid filenames saved!")

# ==========================================
# 5. Age Statistics
# ==========================================

print("\nAge Statistics")
print("-" * 55)

print(df["age"].describe())

print("\nMinimum age:", df["age"].min())
print("Maximum age:", df["age"].max())
print("Mean age:", round(df["age"].mean(), 2))
print("Median age:", df["age"].median())

# ==========================================
# 6. Age Distribution
# ==========================================

plt.figure(figsize=(14, 6))

sns.histplot(
    data=df,
    x="age",
    bins=100,
    kde=True
)

plt.title("UTKFace Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Images")
plt.tight_layout()

plt.savefig(
    "age_distribution.png",
    dpi=300
)

plt.show()

# ==========================================
# 7. Age Group Analysis
# ==========================================

age_bins = [0, 4, 12, 18, 30, 45, 60, 75, 120]
age_labels = [
    "0-3",
    "4-11",
    "12-17",
    "18-29",
    "30-44",
    "45-59",
    "60-74",
    "75+"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)

age_group_counts = df["age_group"].value_counts().sort_index()

print("\nAge Group Distribution")
print("-" * 55)

print(age_group_counts)

# ==========================================
# 8. Save Dataset Report
# ==========================================

df.to_csv(
    "utkface_metadata.csv",
    index=False
)

print("\nMetadata saved successfully!")
print("Age distribution plot saved successfully!")

print("\nExploration completed!")