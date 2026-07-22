# =========================================================
# DATASET PATHS
# =========================================================

TRAIN_DIR = r"D:\MSSA_Net_Project\data\wavelet_dataset\train"

VAL_DIR = r"D:\MSSA_Net_Project\data\wavelet_dataset\val"

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4

# =========================================================
# TRAINING SETTINGS
# =========================================================

EPOCHS = 20

NUM_CLASSES = 2

LSTM_UNITS = 128

DROPOUT_RATE = 0.5

CLASS_NAMES = ['non_seizure', 'seizure']

# =========================================================
# MODEL SAVE PATH
# =========================================================

MODEL_SAVE_PATH = r"D:\MSSA_Net_Project\outputs\models\mssa_net.keras"