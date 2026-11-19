
from pathlib import Path
import hashlib

folder1 = Path("dataset/UTKFace")
folder2 = Path("dataset/crop_part1")


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


print("Scanning UTKFace...")

hashes = {}

for file_path in folder1.iterdir():
    if file_path.is_file():
        file_hash = calculate_hash(file_path)
        hashes.setdefault(file_hash, []).append(file_path.name)

print("UTKFace images:", sum(map(len, hashes.values())))

print("\nScanning crop_part1...")

checked = 0
duplicate_count = 0
same_name_count = 0

for file_path in folder2.iterdir():

    if not file_path.is_file():
        continue

    checked += 1
    file_hash = calculate_hash(file_path)

    if file_hash in hashes:
        duplicate_count += 1

        if file_path.name in hashes[file_hash]:
            same_name_count += 1

print("\n" + "=" * 45)
print("Cross-folder duplicate analysis")
print("=" * 45)

print("Images checked:", checked)
print("Identical image contents:", duplicate_count)
print("Identical content and filename:", same_name_count)
print("Unique images across folders:",
      sum(map(len, hashes.values())) + checked - duplicate_count)

print("\nAnalysis completed!")