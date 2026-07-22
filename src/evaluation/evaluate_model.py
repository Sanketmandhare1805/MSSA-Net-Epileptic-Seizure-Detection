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

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

import tensorflow as tf

from src.utils.config import *

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

RESULTS_DIR = r"D:\MSSA_Net_Project\outputs\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

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
# LOAD MODEL
# =========================================================

print("\nLoading trained MSSA-Net model...\n")

model = tf.keras.models.load_model(
    MODEL_SAVE_PATH
)

# =========================================================
# TRUE LABELS
# =========================================================

y_true = []

for images, labels in val_dataset:

    y_true.extend(
        np.argmax(labels.numpy(), axis=1)
    )

y_true = np.array(y_true)

# =========================================================
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...\n")

y_probs = model.predict(val_dataset)

y_pred = np.argmax(y_probs, axis=1)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASS_NAMES

)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix")

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

)

plt.close()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES

)

print("\nClassification Report:\n")

print(report)

with open(

    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probs[:,1]
)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    )

)

plt.close()

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\nEvaluation completed successfully.\n")

print("Results saved in:")

print(RESULTS_DIR)