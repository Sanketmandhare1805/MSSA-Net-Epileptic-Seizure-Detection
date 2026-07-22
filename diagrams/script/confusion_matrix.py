import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# Paths
# =====================================================

MODEL_PATH = r"D:\MSSA_Net_Project\outputs\models\final_mssa_net.keras"

VALIDATION_PATH = r"D:\MSSA_Net_Project\data\wavelet_dataset\val"

SAVE_PATH = r"D:\MSSA_Net_Project\diagrams\figure\confusion_matrix.png"

# =====================================================
# Parameters
# =====================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

# =====================================================
# Load Model
# =====================================================

print("\nLoading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# Data Generator
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
# True Labels
# =====================================================

y_true = validation_generator.classes

# =====================================================
# Predictions
# =====================================================

print("\nGenerating predictions...")

y_pred_prob = model.predict(validation_generator)

# Softmax handling
if y_pred_prob.shape[1] == 2:
    y_pred_prob = y_pred_prob[:,1]

else:
    y_pred_prob = y_pred_prob.ravel()

# =====================================================
# Convert Probabilities to Labels
# =====================================================

threshold = 0.5

y_pred = (y_pred_prob >= threshold).astype(int)

# =====================================================
# Compute Confusion Matrix
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# Plot
# =====================================================

fig, ax = plt.subplots(figsize=(6,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Non-Seizure', 'Seizure']
)

disp.plot(
    cmap='Blues',
    ax=ax,
    colorbar=False
)

plt.title(
    'Confusion Matrix for MSSA-Net',
    fontsize=16,
    fontweight='bold'
)

# =====================================================
# Save Figure
# =====================================================

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches='tight'
)

print("\nConfusion matrix saved successfully:")
print(SAVE_PATH)

plt.show()