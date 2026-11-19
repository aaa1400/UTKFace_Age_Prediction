
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================
# 1. Configuration
# ==========================================

RANDOM_STATE = 42

METADATA_FILE = Path("utkface_metadata.csv")
OUTPUT_DIR = Path("splits")

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

df = pd.read_csv(METADATA_FILE)

print("=" * 55)
print("UTKFace Dataset Splitting")
print("=" * 55)

print("\nTotal images:", len(df))

# ==========================================
# 3. Create Age Groups
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

# Add image path
df["image_path"] = df["filename"].apply(
    lambda name: str(Path("dataset") / "UTKFace" / name)
)

# ==========================================
# 4. Split: 70% Train, 30% Temporary
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["age_group"]
)

# ==========================================
# 5. Split Temporary: 15% Validation, 15% Test
# ==========================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["age_group"]
)

# ==========================================
# 6. Save Split Files
# ==========================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# ==========================================
# 7. Display Split Summary
# ==========================================

datasets = {
    "Train": train_df,
    "Validation": val_df,
    "Test": test_df
}

print("\nDataset Split Summary")
print("-" * 55)

for name, data in datasets.items():

    print(f"\n{name}:")
    print("Images:", len(data))
    print("Percentage:", round(
        len(data) / len(df) * 100, 2
    ))

# ==========================================
# 8. Compare Age Distribution
# ==========================================

print("\nAge Group Distribution by Dataset")
print("-" * 55)

for name, data in datasets.items():

    print(f"\n{name}:")

    distribution = (
        data["age_group"]
        .value_counts(normalize=True)
        .reindex(age_labels)
        * 100
    )

    print(distribution.round(2))

# ==========================================
# 9. Final Verification
# ==========================================

total_split = (
    len(train_df) +
    len(val_df) +
    len(test_df)
)

print("\n" + "=" * 55)

print("Original Dataset:", len(df))
print("Total Split Images:", total_split)

print("Split Verification:",
      "SUCCESS" if total_split == len(df) else "FAILED")

print("\nSplit files saved in:", OUTPUT_DIR)

print("\nDataset splitting completed!")