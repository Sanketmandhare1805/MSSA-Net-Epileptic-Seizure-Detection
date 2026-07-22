import os
import random
import shutil

# =========================================================
# SOURCE DATASET
# =========================================================

SOURCE_DIR = r"D:\MSSA_Net_Project\data\wavelet_scalograms"

# =========================================================
# TARGET DATASET
# =========================================================

TARGET_DIR = r"D:\MSSA_Net_Project\data\wavelet_dataset"

TRAIN_DIR = os.path.join(TARGET_DIR, "train")

VAL_DIR = os.path.join(TARGET_DIR, "val")

CLASSES = ["seizure", "non_seizure"]

# =========================================================
# CREATE DIRECTORIES
# =========================================================

for split in [TRAIN_DIR, VAL_DIR]:

    for cls in CLASSES:

        os.makedirs(
            os.path.join(split, cls),
            exist_ok=True
        )

# =========================================================
# SPLIT RATIO
# =========================================================

TRAIN_RATIO = 0.8

# =========================================================
# PROCESS EACH CLASS
# =========================================================

for cls in CLASSES:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        cls
    )

    files = os.listdir(source_class_dir)

    random.shuffle(files)

    split_index = int(
        len(files) * TRAIN_RATIO
    )

    train_files = files[:split_index]

    val_files = files[split_index:]

    # =============================================
    # COPY TRAIN FILES
    # =============================================

    for file in train_files:

        src_path = os.path.join(
            source_class_dir,
            file
        )

        dst_path = os.path.join(
            TRAIN_DIR,
            cls,
            file
        )

        shutil.copy(src_path, dst_path)

    # =============================================
    # COPY VAL FILES
    # =============================================

    for file in val_files:

        src_path = os.path.join(
            source_class_dir,
            file
        )

        dst_path = os.path.join(
            VAL_DIR,
            cls,
            file
        )

        shutil.copy(src_path, dst_path)

    print(f"\n{cls}")

    print(f"Train: {len(train_files)}")

    print(f"Validation: {len(val_files)}")

print("\nDataset split completed successfully.\n")