import tensorflow as tf

from tensorflow.keras import layers

from src.utils.config import *


def bilstm_block(inputs):

    # =====================================================
    # RESHAPE FEATURE MAPS INTO SEQUENCE
    # =====================================================

    shape = inputs.shape

    x = layers.Reshape(
        (shape[1] * shape[2], shape[3])
    )(inputs)

    # =====================================================
    # BIDIRECTIONAL LSTM
    # =====================================================

    x = layers.Bidirectional(

        layers.LSTM(
            LSTM_UNITS,
            return_sequences=False
        )

    )(x)

    # =====================================================
    # DROPOUT
    # =====================================================

    x = layers.Dropout(DROPOUT_RATE)(x)

    return x