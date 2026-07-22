import os
import re
import numpy as np
import pywt
import mne

from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================

BASE_RAW_DIR = r"D:\Seizure_Detection_Project\data\raw\chb-mit"

OUTPUT_DIR = r"D:\MSSA_Net_Project\data\wavelet_scalograms"

SEIZURE_DIR = os.path.join(OUTPUT_DIR, "seizure")
NON_SEIZURE_DIR = os.path.join(OUTPUT_DIR, "non_seizure")

os.makedirs(SEIZURE_DIR, exist_ok=True)
os.makedirs(NON_SEIZURE_DIR, exist_ok=True)

# =========================================================
# PARAMETERS
# =========================================================

WINDOW_SIZE = 5

SAMPLING_RATE = 256

TARGET_SIZE = (224, 224)

MAX_IMAGES_PER_CLASS = 5000

# =========================================================
# EEG CHANNELS
# =========================================================

SELECTED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7"
]

# =========================================================
# WAVELET SETTINGS
# =========================================================

SCALES = np.arange(1, 128)

WAVELET = 'morl'

# =========================================================
# GENERATE WAVELET IMAGE
# =========================================================

def generate_wavelet(signal):

    coeffs, freqs = pywt.cwt(
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

    coeffs = (coeffs * 255).astype(np.uint8)

    return coeffs

# =========================================================
# CREATE RGB SCALOGRAM
# =========================================================

def create_rgb_scalogram(signals):

    images = []

    for signal in signals:

        wavelet_img = generate_wavelet(signal)

        img = Image.fromarray(wavelet_img)

        img = img.resize(TARGET_SIZE)

        images.append(np.array(img))

    rgb_image = np.stack(images, axis=-1)

    return rgb_image

# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(image, path):

    img = Image.fromarray(image)

    img.save(path)

# =========================================================
# PARSE SUMMARY FILE
# =========================================================

def parse_summary(summary_path):

    seizure_info = []

    with open(summary_path, 'r') as f:

        lines = f.readlines()

    current_file = None

    seizure_start = None

    seizure_end = None

    for line in lines:

        if "File Name:" in line:

            current_file = line.split(":")[1].strip()

        if "Seizure Start Time:" in line:

            seizure_start = int(
                re.findall(r'\d+', line)[0]
            )

        if "Seizure End Time:" in line:

            seizure_end = int(
                re.findall(r'\d+', line)[0]
            )

            seizure_info.append({
                "file": current_file,
                "start": seizure_start,
                "end": seizure_end
            })

    return seizure_info

# =========================================================
# PROCESS SEIZURE WINDOWS
# =========================================================

def process_seizure_file(
    edf_path,
    seizure_start,
    seizure_end,
    image_counter
):

    try:

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False
        )

        data = raw.get_data(
            picks=SELECTED_CHANNELS
        )

        start_sample = seizure_start * SAMPLING_RATE

        end_sample = seizure_end * SAMPLING_RATE

        window_samples = WINDOW_SIZE * SAMPLING_RATE

        current = start_sample

        while current + window_samples < end_sample:

            if image_counter >= MAX_IMAGES_PER_CLASS:
                break

            signals = []

            for ch in range(3):

                signal = data[ch, current:current+window_samples]

                signals.append(signal)

            rgb_image = create_rgb_scalogram(signals)

            save_path = os.path.join(
                SEIZURE_DIR,
                f"seizure_{image_counter}.png"
            )

            save_image(rgb_image, save_path)

            image_counter += 1

            current += window_samples // 2

        return image_counter

    except Exception as e:

        print(f"Error: {edf_path}")

        print(e)

        return image_counter

# =========================================================
# PROCESS NON-SEIZURE WINDOWS
# =========================================================

def process_non_seizure_file(
    edf_path,
    image_counter
):

    try:

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False
        )

        data = raw.get_data(
            picks=SELECTED_CHANNELS
        )

        total_samples = data.shape[1]

        window_samples = WINDOW_SIZE * SAMPLING_RATE

        current = 0

        while current + window_samples < total_samples:

            if image_counter >= MAX_IMAGES_PER_CLASS:
                break

            signals = []

            for ch in range(3):

                signal = data[ch, current:current+window_samples]

                signals.append(signal)

            rgb_image = create_rgb_scalogram(signals)

            save_path = os.path.join(
                NON_SEIZURE_DIR,
                f"non_seizure_{image_counter}.png"
            )

            save_image(rgb_image, save_path)

            image_counter += 1

            current += window_samples * 30

        return image_counter

    except Exception as e:

        print(f"Error: {edf_path}")

        print(e)

        return image_counter

# =========================================================
# MAIN PIPELINE
# =========================================================

print("\nStarting seizure-aware wavelet generation...\n")

seizure_count = 0

non_seizure_count = 0

for patient in os.listdir(BASE_RAW_DIR):

    patient_path = os.path.join(
        BASE_RAW_DIR,
        patient
    )

    if not os.path.isdir(patient_path):
        continue

    summary_file = os.path.join(
        patient_path,
        f"{patient}-summary.txt"
    )

    if not os.path.exists(summary_file):
        continue

    seizure_data = parse_summary(summary_file)

    # =============================================
    # PROCESS SEIZURE FILES
    # =============================================

    for seizure in seizure_data:

        edf_path = os.path.join(
            patient_path,
            seizure["file"]
        )

        seizure_count = process_seizure_file(
            edf_path,
            seizure["start"],
            seizure["end"],
            seizure_count
        )

    # =============================================
    # PROCESS NON-SEIZURE FILES
    # =============================================

    for file in os.listdir(patient_path):

        if not file.endswith(".edf"):
            continue

        edf_path = os.path.join(
            patient_path,
            file
        )

        non_seizure_count = process_non_seizure_file(
            edf_path,
            non_seizure_count
        )

    if (
        seizure_count >= MAX_IMAGES_PER_CLASS
        and
        non_seizure_count >= MAX_IMAGES_PER_CLASS
    ):
        break

print("\nWavelet dataset generation completed.\n")

print(f"Seizure images: {seizure_count}")

print(f"Non-seizure images: {non_seizure_count}")