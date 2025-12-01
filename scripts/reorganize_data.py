import os
import shutil
from pathlib import Path

print("=" * 70)
print("REORGANIZING DATA INTO BINARY CLASSIFICATION")
print("=" * 70)

# Source directory (old project)
source_dir = Path.home() / "dual_eye_ai_OLD_BACKUP" / "data_balanced"

# Destination directory (new project)
dest_dir = Path("dataset")

# Class mapping
normal_classes = ["Normal"]
abnormal_classes = ["Allergies", "Conjunctivitis", "Eyelid_Issues", "Redness"]

# Copy Normal images
print("\nCopying NORMAL images...")
normal_dest = dest_dir / "Normal"
normal_count = 0

for class_name in normal_classes:
    source_class = source_dir / class_name
    if source_class.exists():
        for img_file in source_class.glob("*.jpg"):
            shutil.copy(img_file, normal_dest / img_file.name)
            normal_count += 1
        for img_file in source_class.glob("*.jpeg"):
            shutil.copy(img_file, normal_dest / img_file.name)
            normal_count += 1
        for img_file in source_class.glob("*.png"):
            shutil.copy(img_file, normal_dest / img_file.name)
            normal_count += 1

print(f"✓ Copied {normal_count} Normal images")

# Copy Abnormal images
print("\nCopying ABNORMAL images...")
abnormal_dest = dest_dir / "Abnormal"
abnormal_count = 0

for class_name in abnormal_classes:
    source_class = source_dir / class_name
    if source_class.exists():
        for img_file in source_class.glob("*.jpg"):
            # Rename to include original class for reference
            new_name = f"{class_name.lower()}_{img_file.name}"
            shutil.copy(img_file, abnormal_dest / new_name)
            abnormal_count += 1
        for img_file in source_class.glob("*.jpeg"):
            new_name = f"{class_name.lower()}_{img_file.name}"
            shutil.copy(img_file, abnormal_dest / new_name)
            abnormal_count += 1
        for img_file in source_class.glob("*.png"):
            new_name = f"{class_name.lower()}_{img_file.name}"
            shutil.copy(img_file, abnormal_dest / new_name)
            abnormal_count += 1

print(f"✓ Copied {abnormal_count} Abnormal images")

print("\n" + "=" * 70)
print("DATA REORGANIZATION COMPLETE")
print("=" * 70)
print(f"\nNormal: {normal_count} images")
print(f"Abnormal: {abnormal_count} images")
print(f"Total: {normal_count + abnormal_count} images")
print(f"\nRatio: {abnormal_count/normal_count:.2f}:1 (Abnormal:Normal)")
print("\n✓ Ready for training!")
