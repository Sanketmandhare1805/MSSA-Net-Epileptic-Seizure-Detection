import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# =====================================================
# PATHS
# =====================================================

DATASET_PATH = r"D:\MSSA_Net_Project\data\bonn_splits\val"

MODEL_PATH = r"D:\MSSA_Net_Project\outputs\models\mssa_net_bonn.keras"

RESULTS_DIR = r"D:\MSSA_Net_Project\outputs\bonn_model_results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Bonn Validation Dataset...\n")

dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=(224, 224),

    batch_size=8,

    shuffle=False

)

class_names = dataset.class_names

print("\nClasses:\n")
print(class_names)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading MSSA-Net Bonn Model...\n")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# =====================================================
# TRUE LABELS
# =====================================================

y_true = []

for images, labels in dataset:

    y_true.extend(
        labels.numpy()
    )

y_true = np.array(
    y_true
)

# =====================================================
# PREDICTIONS
# =====================================================

print("\nGenerating Predictions...\n")

pred_probs = model.predict(
    dataset
)

y_pred = np.argmax(
    pred_probs,
    axis=1
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names

    )

)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Bonn Dataset Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center"

        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_DIR,
    "bonn_confusion_matrix.png"
)

plt.savefig(
    confusion_path
)

plt.show()

# =====================================================
# ROC CURVE
# =====================================================

fpr, tpr, thresholds = roc_curve(

    y_true,

    pred_probs[:, 1]

)

roc_auc = auc(
    fpr,
    tpr
)
print("\nAUC SCORE:")
print(roc_auc)
plt.figure(figsize=(7, 6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.4f}"

)

plt.plot(
    [0, 1],
    [0, 1],
    '--'
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Bonn Dataset ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "bonn_roc_curve.png"
)

plt.savefig(
    roc_path
)

plt.show()

# =====================================================
# SAVE REPORT
# =====================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    f.write("\n\n")

    f.write(
        f"AUC Score : {roc_auc:.4f}\n"
    )

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n===================================")
print(" BONN MODEL EVALUATION COMPLETED ")
print("===================================")

print(f"\nAUC Score : {roc_auc:.4f}")

print("\nResults Saved In:")

print(RESULTS_DIR)

print("\nGenerated Files:")

print("1. bonn_confusion_matrix.png")
print("2. bonn_roc_curve.png")
print("3. classification_report.txt")