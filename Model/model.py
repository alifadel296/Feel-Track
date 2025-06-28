from tensorflow.keras.layers import (
    LSTM,
    Attention,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)

from tensorflow.keras.models import Model


def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN Blocks
    x = Conv1D(128, 5, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Attention mechanism
    x = Attention()([x, x])

    # Dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)
