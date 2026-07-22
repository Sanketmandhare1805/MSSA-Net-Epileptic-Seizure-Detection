import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from src.models.mssa_net import build_mssa_net

from src.utils.config import *

# =========================================================
# LOAD TRAIN DATASET
# =========================================================

print("\nLoading training dataset...\n")

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode='categorical',

    shuffle=True

)

# =========================================================
# LOAD VALIDATION DATASET
# =========================================================

print("\nLoading validation dataset...\n")

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(

    VAL_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode='categorical',

    shuffle=False

)

# =========================================================
# PREFETCH
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# =========================================================
# BUILD MODEL
# =========================================================

print("\nBuilding MSSA-Net model...\n")

model = build_mssa_net()

# =========================================================
# CALLBACKS
# =========================================================

checkpoint = ModelCheckpoint(

    MODEL_SAVE_PATH,

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor='val_accuracy',

    patience=5,

    restore_best_weights=True

)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nStarting MSSA-Net training...\n")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=[checkpoint, early_stop]

)

print("\nTraining completed successfully.\n")