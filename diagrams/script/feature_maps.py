import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image

# =====================================================
# Paths
# =====================================================

MODEL_PATH = r"D:\MSSA_Net_Project\outputs\models\final_mssa_net.keras"

IMAGE_PATH = r"D:\MSSA_Net_Project\data\wavelet_dataset\val\seizure"

SAVE_PATH = r"D:\MSSA_Net_Project\diagrams\figure\feature_maps.png"

# =====================================================
# Load Model
# =====================================================

print("\nLoading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# Select First Conv Layer
# =====================================================

layer_name = "conv2d"

feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer(layer_name).output
)

# =====================================================
# Load One Image
# =====================================================

import os

image_name = os.listdir(IMAGE_PATH)[0]

img_path = os.path.join(
    IMAGE_PATH,
    image_name
)

print("\nUsing image:")
print(img_path)

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# =====================================================
# Predict Feature Maps
# =====================================================

feature_maps = feature_model.predict(img_array)

# =====================================================
# Plot First 16 Feature Maps
# =====================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(10,10)
)

fig.suptitle(
    'Feature Map Visualization',
    fontsize=18,
    fontweight='bold'
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        feature_maps[0, :, :, i],
        cmap='viridis'
    )

    ax.axis('off')

    ax.set_title(f'Filter {i+1}')

# =====================================================
# Layout
# =====================================================

plt.tight_layout()

# =====================================================
# Save Figure
# =====================================================

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches='tight'
)

print("\nFeature map figure saved successfully:")
print(SAVE_PATH)

plt.show()