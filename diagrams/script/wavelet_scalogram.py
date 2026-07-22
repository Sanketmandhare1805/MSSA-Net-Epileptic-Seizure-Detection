import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne

# =====================================================
# Load EEG EDF File
# =====================================================

file_path = r"D:\Seizure_Detection_Project\data\raw\chb-mit\chb01\chb01_03.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)
print(raw.ch_names)
# =====================================================
# Select EEG Channel
# =====================================================

channel_name = 'FP1-F7'

signal = raw.get_data(picks=channel_name)[0]

sampling_rate = int(raw.info['sfreq'])

# =====================================================
# Select EEG Window
# =====================================================

start_sample = 10000
end_sample = 20000

signal = signal[start_sample:end_sample]

time = np.arange(len(signal)) / sampling_rate

# =====================================================
# Continuous Wavelet Transform
# =====================================================

scales = np.arange(1, 128)

coefficients, frequencies = pywt.cwt(
    signal,
    scales,
    'morl',
    sampling_period=1/sampling_rate
)

# =====================================================
# Create Figure
# =====================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)

# -----------------------------------------------------
# EEG Signal Plot
# -----------------------------------------------------

axes[0].plot(
    time,
    signal,
    linewidth=1
)

axes[0].set_title(
    "EEG Signal",
    fontsize=14,
    fontweight='bold'
)

axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

axes[0].grid(True, alpha=0.3)

# -----------------------------------------------------
# Wavelet Scalogram Plot
# -----------------------------------------------------

im = axes[1].imshow(
    np.abs(coefficients),
    extent=[
        time.min(),
        time.max(),
        frequencies.max(),
        frequencies.min()
    ],
    cmap='jet',
    aspect='auto'
)

axes[1].set_title(
    "Wavelet Scalogram (CWT)",
    fontsize=14,
    fontweight='bold'
)

axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Frequency (Hz)")

# Colorbar
cbar = fig.colorbar(
    im,
    ax=axes[1]
)

cbar.set_label('Magnitude')

# =====================================================
# Layout
# =====================================================

plt.tight_layout()

# =====================================================
# Save Figure
# =====================================================

save_path = r"D:\MSSA_Net_Project\diagrams\figure\wavelet_scalogram.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

print("\nFigure saved successfully:")
print(save_path)

plt.show()