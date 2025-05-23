import tensorflow as tf
from tensorflow.keras import layers, Model

def EfficientNetB1(shape1, shape2, shape3, num_classes):
    input_shape=(shape1,shape2,shape3)
    # Create input layer for single-channel input
    efficientNet = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights=None,  # Set to None because weights are for 3-channel inputs
        input_shape=input_shape
    )
    input_layer = layers.Input(shape=input_shape)

    # Pass through EfficientNet
    x = efficientNet(input_layer)

    # Decoder to upsample feature maps
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)

    # Final layer for 6 classes
    output_layer = layers.Conv2DTranspose(num_classes, (3, 3), strides=(2, 2), padding='same', activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model