"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.applications

from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from Squeeze_and_Excite import Squeeze_and_Excite


def identity_block(input_tensor, kernel_size, filters, stage, block, squeeze=False, squeeze_type='normal'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters
    # K.learning_phase()
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # squeeze_block = Squeeze_and_Excite(input_tensor.get_shape()[bn_axis])
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if squeeze == True and squeeze_type == 'pre':
        squeeze_block = Squeeze_and_Excite(input_tensor.get_shape()[bn_axis])
        x = squeeze_block(input_tensor)

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # K.int_shape(input_tensor)[bn_axis]

    if squeeze == True and squeeze_type == 'normal':
        squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
        x = squeeze_block(x)

    if squeeze_type != 'identity':  # Never have squeeze = False and squeeze_type = 'identity'
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)

    if squeeze == True and squeeze_type == 'post':
        squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
        x = squeeze_block(x)

    if squeeze == True and squeeze_type == 'identity':
        squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
        y = squeeze_block(input_tensor)
        x = layers.add([y, x])

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), squeeze=False, squeeze_type='normal'):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    """
    tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    dilation_rate=(1, 1), activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs
    )
    """

    filters1, filters2, filters3 = filters
    # K.learning_phase()
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if squeeze == True and squeeze_type == 'pre':
        squeeze_block = Squeeze_and_Excite(input_tensor.get_shape()[bn_axis])
        x = squeeze_block(input_tensor)

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    # shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut, training = False)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    if squeeze == True and squeeze_type == 'normal':
        squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
        x = squeeze_block(x)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    if squeeze == True and squeeze_type == 'post':
        squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
        x = squeeze_block(x)

    # if squeeze == True and squeeze_type == 'identity':
    # squeeze_block = Squeeze_and_Excite(x.get_shape()[bn_axis])
    # y = squeeze_block(input_tensor)
    # x = layers.add([y, x])

    return x


# Modification to CIFAR10
def ResNet50(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=10, squeeze=False,
             squeeze_type='normal', **kwargs):
    """Instantiates the ResNet50 architecture.
    
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # Determine proper input shape
    input_shape = (32, 32, 3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # K.learning_phase()
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    # x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x, training = False)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', squeeze=squeeze, squeeze_type=squeeze_type)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', squeeze=squeeze, squeeze_type=squeeze_type)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', squeeze=squeeze, squeeze_type=squeeze_type)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', squeeze=squeeze, squeeze_type=squeeze_type)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', squeeze=squeeze, squeeze_type=squeeze_type)

    # Output shape: (1, 1, depth)

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # print("Output shape :")
    # print(x.get_shape())

    if include_top:
        x = Flatten()(x)
        # print("After flatten ")
        # print(x.get_shape())
        x = Dense(classes, activation='softmax', name='fc1000')(x)
        # print("After Dense ")
        # print(x.get_shape())

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    return model


if __name__ == '__main__':
    model = ResNet50(include_top=True, weights=None, squeeze=False, squeeze_type='Normal')

    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
