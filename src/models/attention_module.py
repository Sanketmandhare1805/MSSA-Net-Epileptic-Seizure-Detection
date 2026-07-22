import tensorflow as tf

from tensorflow.keras import layers


def seizure_attention_module(inputs):

    # =====================================================
    # CHANNEL ATTENTION
    # =====================================================

    avg_pool = layers.GlobalAveragePooling2D()(inputs)

    dense_1 = layers.Dense(
        inputs.shape[-1] // 2,
        activation='relu'
    )(avg_pool)

    dense_2 = layers.Dense(
        inputs.shape[-1],
        activation='sigmoid'
    )(dense_1)

    channel_attention = layers.Reshape(
        (1, 1, inputs.shape[-1])
    )(dense_2)

    # =====================================================
    # APPLY ATTENTION
    # =====================================================

    output = layers.Multiply()([
        inputs,
        channel_attention
    ])

    return output