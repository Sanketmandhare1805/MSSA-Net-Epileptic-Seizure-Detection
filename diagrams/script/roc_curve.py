import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# Paths
# =====================================================

MODEL_PATH = r"D:\MSSA_Net_Project\outputs\models\mssa_net.keras"

VALIDATION_PATH = r"D:\MSSA_Net_Project\data\wavelet_dataset\val"

SAVE_PATH = r"D:\MSSA_Net_Project\diagrams\figure\roc_curve.png"

# =====================================================
# Image Parameters
# =====================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

# =====================================================
# Load Model
# =====================================================

print("\nLoading MSSA-Net model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# Validation Data Generator
# =====================================================

datagen = ImageDataGenerator()

validation_generator = datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =====================================================
# Print Class Mapping
# =====================================================

print("\nClass Indices:")
print(validation_generator.class_indices)

# =====================================================
# True Labels
# =====================================================

y_true = validation_generator.classes

# =====================================================
# Predict Probabilities
# =====================================================

print("\nGenerating predictions...")

y_pred_prob = model.predict(validation_generator)

# =====================================================
# Handle Softmax Output
# =====================================================

if y_pred_prob.shape[1] == 2:
    # Use seizure class probability
    y_pred_prob = y_pred_prob[:, 1]

else:
    # Sigmoid output
    y_pred_prob = y_pred_prob.ravel()

# =====================================================
# Compute ROC
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_pred_prob
)

roc_auc = auc(fpr, tpr)

print(f"\nAUC Score: {roc_auc:.4f}")

# =====================================================
# Plot ROC Curve
# =====================================================

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'MSSA-Net (AUC = {roc_auc:.4f})'
)

# Random classifier line
plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=1
)

# =====================================================
# Labels
# =====================================================

plt.title(
    'ROC Curve for MSSA-Net',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel(
    'False Positive Rate',
    fontsize=12
)

plt.ylabel(
    'True Positive Rate',
    fontsize=12
)

plt.legend(
    loc='lower right'
)

plt.grid(alpha=0.3)

# =====================================================
# Save Figure
# =====================================================

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches='tight'
)

print("\nROC curve saved successfully:")
print(SAVE_PATH)

plt.show()