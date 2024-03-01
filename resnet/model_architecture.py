from tensorflow.keras import layers, models

def build_1d_resnet(input_shape, num_classes):
    input_tensor = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks
    x = residual_block_1d(x, 64, 1)
    x = residual_block_1d(x, 128, 2)
    x = residual_block_1d(x, 256, 2)
    x = residual_block_1d(x, 512, 2)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=input_tensor, outputs=x, name='resnet_1d')

    return model

def residual_block_1d(input_tensor, filters, strides):
    shortcut = input_tensor

    # First convolution layer
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same', activation='relu')(input_tensor)

    # Second convolution layer
    x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)

    # Shortcut connection if needed
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='valid', activation='relu')(input_tensor)

    # Add shortcut to main path
    x = layers.add([x, shortcut])

    return x