import os
import random
import shutil

# =====================================================
# PATHS
# =====================================================

SOURCE_DIR = r"D:\MSSA_Net_Project\data\bonn_wavelet"

OUTPUT_DIR = r"D:\MSSA_Net_Project\data\bonn_splits"

TRAIN_RATIO = 0.8

random.seed(42)

# =====================================================
# CREATE FOLDERS
# =====================================================

for split in ["train", "val"]:

    for cls in ["seizure", "non_seizure"]:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                cls
            ),
            exist_ok=True
        )

# =====================================================
# SPLIT DATA
# =====================================================

for cls in ["seizure", "non_seizure"]:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        cls
    )

    files = os.listdir(
        source_class_dir
    )

    random.shuffle(files)

    split_index = int(
        len(files) * TRAIN_RATIO
    )

    train_files = files[:split_index]

    val_files = files[split_index:]

    # Train

    for file in train_files:

        shutil.copy2(

            os.path.join(
                source_class_dir,
                file
            ),

            os.path.join(
                OUTPUT_DIR,
                "train",
                cls,
                file
            )

        )

    # Validation

    for file in val_files:

        shutil.copy2(

            os.path.join(
                source_class_dir,
                file
            ),

            os.path.join(
                OUTPUT_DIR,
                "val",
                cls,
                file
            )

        )

    print(f"\n{cls}")

    print(
        f"Train: {len(train_files)}"
    )

    print(
        f"Validation: {len(val_files)}"
    )

print(
    "\nBonn dataset split completed."
)