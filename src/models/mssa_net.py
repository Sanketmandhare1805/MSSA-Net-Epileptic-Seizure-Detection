import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from tensorflow.keras import layers
from tensorflow.keras import models

from src.models.multiscale_block import multi_scale_block
from src.models.attention_module import seizure_attention_module
from src.models.bilstm_block import bilstm_block

from src.utils.config import *

# =========================================================
# BUILD MSSA-NET
# =========================================================

def build_mssa_net():

    # =====================================================
    # INPUT LAYER
    # =====================================================

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    x = layers.Rescaling(1./255)(inputs)

    # =====================================================
    # MULTI-SCALE BLOCK 1
    # =====================================================

    x = multi_scale_block(x)
    

    x = layers.MaxPooling2D(
        pool_size=(2,2)
    )(x)

    # =====================================================
    # ATTENTION MODULE 1
    # =====================================================

    x = seizure_attention_module(x)

    # =====================================================
    # MULTI-SCALE BLOCK 2
    # =====================================================

    x = multi_scale_block(x)

    x = layers.MaxPooling2D(
        pool_size=(2,2)
    )(x)

    # =====================================================
    # ATTENTION MODULE 2
    # =====================================================

    x = seizure_attention_module(x)

    # =====================================================
    # CNN FEATURE EXTRACTION
    # =====================================================

    x = layers.Conv2D(
        128,
        (3,3),
        padding='same',
        activation='relu'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2,2)
    )(x)

    # =====================================================
    # BiLSTM BLOCK
    # =====================================================

    x = bilstm_block(x)

    # =====================================================
    # FULLY CONNECTED LAYERS
    # =====================================================

    x = layers.Dense(
        256,
        activation='relu'
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE
    )(x)

    # =====================================================
    # OUTPUT LAYER
    # =====================================================

    outputs = layers.Dense(
        NUM_CLASSES,
        activation='softmax'
    )(x)

    # =====================================================
    # CREATE MODEL
    # =====================================================

    model = models.Model(
        inputs,
        outputs
    )

    # =====================================================
    # COMPILE MODEL
    # =====================================================

    model.compile(

        optimizer='adam',

        loss='categorical_crossentropy',

        metrics=['accuracy']

    )

    return model