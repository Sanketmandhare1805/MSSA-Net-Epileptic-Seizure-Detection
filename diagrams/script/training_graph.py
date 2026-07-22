
import matplotlib.pyplot as plt

# =====================================================
# Epochs
# =====================================================

epochs = list(range(1, 15))

# =====================================================
# Training Accuracy
# =====================================================

train_acc = [
    0.6733,
    0.7564,
    0.8063,
    0.8278,
    0.8793,
    0.8804,
    0.8889,
    0.9024,
    0.9174,
    0.9292,
    0.9303,
    0.9447,
    0.9571,
    0.9544
]

# =====================================================
# Validation Accuracy
# =====================================================

val_acc = [
    0.8155,
    0.7833,
    0.8176,
    0.8648,
    0.8863,
    0.8584,
    0.8863,
    0.8777,
    0.9163,
    0.9120,
    0.8391,
    0.8734,
    0.8648,
    0.8755
]

# =====================================================
# Training Loss
# =====================================================

train_loss = [
    0.6355,
    0.5186,
    0.4341,
    0.3966,
    0.3088,
    0.3197,
    0.2833,
    0.2476,
    0.2147,
    0.1890,
    0.1862,
    0.1722,
    0.1374,
    0.1211
]

# =====================================================
# Validation Loss
# =====================================================

val_loss = [
    0.4810,
    0.4465,
    0.4260,
    0.3468,
    0.3248,
    0.3392,
    0.3072,
    0.3267,
    0.2896,
    0.2626,
    0.3738,
    0.3812,
    0.3942,
    0.4718
]

# =====================================================
# Plot Accuracy
# =====================================================

plt.figure(figsize=(10,6))

plt.plot(
    epochs,
    train_acc,
    marker='o',
    linewidth=2,
    label='Training Accuracy'
)

plt.plot(
    epochs,
    val_acc,
    marker='s',
    linewidth=2,
    label='Validation Accuracy'
)

# =====================================================
# Labels
# =====================================================

plt.title(
    'Training and Validation Accuracy',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.grid(alpha=0.3)

# =====================================================
# Save Accuracy Figure
# =====================================================

accuracy_path = r"D:\MSSA_Net_Project\diagrams\figure\training_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches='tight'
)

print("\nAccuracy graph saved successfully:")
print(accuracy_path)

plt.show()

# =====================================================
# Plot Loss
# =====================================================

plt.figure(figsize=(10,6))

plt.plot(
    epochs,
    train_loss,
    marker='o',
    linewidth=2,
    label='Training Loss'
)

plt.plot(
    epochs,
    val_loss,
    marker='s',
    linewidth=2,
    label='Validation Loss'
)

# =====================================================
# Labels
# =====================================================

plt.title(
    'Training and Validation Loss',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.grid(alpha=0.3)

# =====================================================
# Save Loss Figure
# =====================================================

loss_path = r"D:\MSSA_Net_Project\diagrams\figure\training_loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches='tight'
)

print("\nLoss graph saved successfully:")
print(loss_path)

plt.show()