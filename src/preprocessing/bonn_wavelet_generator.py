import os
import numpy as np
import pandas as pd
import pywt

from PIL import Image
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

CSV_PATH = r"D:\Base_Paper_Projet\data\raw\Epileptic Seizure Recognition.csv"

OUTPUT_DIR = r"D:\MSSA_Net_Project\data\bonn_wavelet_v2"

SEIZURE_DIR = os.path.join(
    OUTPUT_DIR,
    "seizure"
)

NON_SEIZURE_DIR = os.path.join(
    OUTPUT_DIR,
    "non_seizure"
)

os.makedirs(SEIZURE_DIR, exist_ok=True)
os.makedirs(NON_SEIZURE_DIR, exist_ok=True)

# =====================================================
# SAME SETTINGS AS CHB-MIT
# =====================================================

TARGET_SIZE = (224, 224)

SCALES = np.arange(1, 128)

WAVELET = "morl"

# =====================================================
# CHB-MIT STYLE WAVELET
# =====================================================

def generate_wavelet(signal):

    coeffs, _ = pywt.cwt(

        signal,

        SCALES,

        WAVELET

    )

    coeffs = np.abs(coeffs)

    coeffs = (

        coeffs - coeffs.min()

    ) / (

        coeffs.max() - coeffs.min() + 1e-8

    )

    coeffs = (

        coeffs * 255

    ).astype(np.uint8)

    return coeffs

# =====================================================
# CREATE 3 PSEUDO EEG CHANNELS
# =====================================================

def create_three_channels(signal):

    # Channel 1
    ch1 = signal.copy()

    # Channel 2 (smoothed)

    kernel = np.ones(5) / 5

    ch2 = np.convolve(

        signal,

        kernel,

        mode='same'

    )

    # Channel 3 (derivative)

    ch3 = np.gradient(signal)

    return [

        ch1,

        ch2,

        ch3

    ]

# =====================================================
# RGB SCALOGRAM
# =====================================================

def create_rgb_scalogram(signal):

    channels = create_three_channels(
        signal
    )

    images = []

    for ch in channels:

        wavelet_img = generate_wavelet(ch)

        img = Image.fromarray(
            wavelet_img
        )

        img = img.resize(
            TARGET_SIZE
        )

        images.append(
            np.array(img)
        )

    rgb_image = np.stack(

        images,

        axis=-1

    )

    return rgb_image

# =====================================================
# SAVE IMAGE
# =====================================================

def save_image(image, path):

    Image.fromarray(
        image
    ).save(path)

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Bonn Dataset...\n")

df = pd.read_csv(
    CSV_PATH
)

df = df.drop(
    columns=["Unnamed"]
)

# =====================================================
# BINARY MAPPING
# =====================================================

seizure_df = df[
    df["y"] == 1
]

non_seizure_df = df[
    df["y"] != 1
]

# Balance

non_seizure_df = non_seizure_df.sample(

    n=len(seizure_df),

    random_state=42

)

print(

    f"Seizure Samples: {len(seizure_df)}"

)

print(

    f"Non-Seizure Samples: {len(non_seizure_df)}"

)

# =====================================================
# SEIZURE IMAGES
# =====================================================

print(

    "\nGenerating seizure images...\n"

)

for idx, row in tqdm(

    seizure_df.iterrows(),

    total=len(seizure_df)

):

    signal = row.drop("y").values.astype(
        np.float32
    )

    rgb_image = create_rgb_scalogram(
        signal
    )

    save_image(

        rgb_image,

        os.path.join(

            SEIZURE_DIR,

            f"seizure_{idx}.png"

        )

    )

# =====================================================
# NON-SEIZURE IMAGES
# =====================================================

print(

    "\nGenerating non-seizure images...\n"

)

for idx, row in tqdm(

    non_seizure_df.iterrows(),

    total=len(non_seizure_df)

):

    signal = row.drop("y").values.astype(
        np.float32
    )

    rgb_image = create_rgb_scalogram(
        signal
    )

    save_image(

        rgb_image,

        os.path.join(

            NON_SEIZURE_DIR,

            f"non_seizure_{idx}.png"

        )

    )

print("\nGeneration Complete.\n")