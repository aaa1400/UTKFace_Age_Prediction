
from pathlib import Path
import hashlib

DATASET_DIR = Path("dataset")

folder_pairs = [
    (
        DATASET_DIR / "UTKFace",
        DATASET_DIR / "utkface_aligned_cropped" / "UTKFace"
    ),
    (
        DATASET_DIR / "crop_part1",
        DATASET_DIR / "utkface_aligned_cropped" / "crop_part1"
    )
]


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


for first_folder, second_folder in folder_pairs:

    print("\n" + "=" * 50)
    print("Folder:", first_folder)

    if not first_folder.exists() or not second_folder.exists():
        print("ERROR: Folder not found!")
        continue

    files = list(first_folder.glob("*"))

    checked = 0
    different = 0
    missing = 0

    for file1 in files:

        if not file1.is_file():
            continue

        file2 = second_folder / file1.name

        if not file2.exists():
            missing += 1
            continue

        hash1 = calculate_hash(file1)
        hash2 = calculate_hash(file2)

        checked += 1

        if hash1 != hash2:
            different += 1

    print("Checked:", checked)
    print("Different images:", different)
    print("Missing images:", missing)

print("\nComparison completed!")