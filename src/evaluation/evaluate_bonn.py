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

DATASET_PATH = r"D:\MSSA_Net_Project\data\bonn_wavelet"

MODEL_PATH = r"D:\MSSA_Net_Project\outputs\models\mssa_net.keras"

RESULTS_DIR = r"D:\MSSA_Net_Project\outputs\final_results\cross_dataset"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Bonn dataset...\n")

dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=(224,224),

    batch_size=32,

    shuffle=False

)

class_names = dataset.class_names

print("\nClasses:\n")

print(class_names)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading MSSA-Net...\n")

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

print("\nGenerating predictions...\n")

pred_probs = model.predict(
    dataset
)

y_pred = np.argmax(
    pred_probs,
    axis=1
)

# =====================================================
# REPORT
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

plt.figure(figsize=(6,5))

plt.imshow(
    cm,
    cmap='Blues'
)

plt.title(
    "Bonn Dataset Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0,1],
    class_names
)

plt.yticks(
    [0,1],
    class_names
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(

            j,
            i,

            str(cm[i,j]),

            ha='center',

            va='center'

        )

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULTS_DIR,

        "bonn_confusion_matrix.png"

    )

)

plt.show()

# =====================================================
# ROC CURVE
# =====================================================

fpr, tpr, _ = roc_curve(

    y_true,

    pred_probs[:,1]

)

roc_auc = auc(
    fpr,
    tpr
)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

    f.write("\n\n")

    f.write(
        f"AUC Score : {roc_auc:.4f}\n"
    )

plt.figure(figsize=(6,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    '--'

)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Bonn ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULTS_DIR,

        "bonn_roc_curve.png"

    )

)

plt.show()

print("\nCross-Dataset Evaluation Completed.\n")

print(

    f"\nAUC Score: {roc_auc:.4f}\n"

)