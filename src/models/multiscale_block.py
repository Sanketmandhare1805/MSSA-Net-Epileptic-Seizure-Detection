import tensorflow as tf
from tensorflow.keras import layers


def multi_scale_block(inputs):

    # =====================================================
    # 3x3 Branch
    # =====================================================

    branch1 = layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        activation='relu'
    )(inputs)

    branch1 = layers.BatchNormalization()(branch1)

    # =====================================================
    # 5x5 Branch
    # =====================================================

    branch2 = layers.Conv2D(
        32,
        (5, 5),
        padding='same',
        activation='relu'
    )(inputs)

    branch2 = layers.BatchNormalization()(branch2)

    # =====================================================
    # 7x7 Branch
    # =====================================================

    branch3 = layers.Conv2D(
        32,
        (7, 7),
        padding='same',
        activation='relu'
    )(inputs)

    branch3 = layers.BatchNormalization()(branch3)

    # =====================================================
    # CONCATENATE FEATURES
    # =====================================================

    output = layers.Concatenate()([
        branch1,
        branch2,
        branch3
    ])

    # =====================================================
    # MAX POOLING
    # =====================================================

    output = layers.MaxPooling2D(pool_size=(2, 2))(output)

    return output