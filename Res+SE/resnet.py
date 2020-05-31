import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


from keras_squeeze_excite_network import TF

if TF:
    from tensorflow.keras import backend as K
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                                         GlobalAveragePooling2D, GlobalMaxPooling2D,
                                         Input, MaxPooling2D, add)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    from tensorflow.keras.backend import is_keras_tensor
    from tensorflow.keras.utils import get_source_inputs
else:
    from keras import backend as K
    from keras.applications.resnet50 import preprocess_input
    from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                              GlobalAveragePooling2D, GlobalMaxPooling2D,
                              Input, MaxPooling2D, add)
    from keras.models import Model
    from keras.regularizers import l2
    from keras.applications.imagenet_utils import decode_predictions
    from keras.utils import get_source_inputs

    is_keras_tensor = K.is_keras_tensor

from keras_squeeze_excite_network.se import squeeze_excite_block
from keras_squeeze_excite_network.utils import _obtain_input_shape, _tensor_shape



class BasicBlock(layers.Layer):
    # 残差类模块
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        # f(x)包含两个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding='same')
        self.bh1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bh2 = layers.BatchNormalization()
        # 插入identity
        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:  # 否者就直接连接
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 前向传播函数
        out = self.conv1(inputs)
        out = self.bh1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bh2(out)
        # 输入通过identity() 转化
        identity = self.downsample(inputs)
        # f(x) + x操作
        output = layers.add([out, identity])
        # 再通过激活函数并返回
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=20):
        super(ResNet, self).__init__()
        #  根网络预处理
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
        # 堆叠4个Block, 每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[0], strides=2)
        self.layer3 = self.build_resblock(256, layer_dims[0], strides=2)
        self.layer4 = self.build_resblock(512, layer_dims[0], strides=2)
        # 通过Pooling层将高管降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs)
        # 一次通过4个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #  通过池化层
        x = self.avgpool(x)
        #  通过全连接层
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, strides=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        # 只有第一个BasicBlock的步长可能不为1
        res_block = Sequential()
        res_block.add(BasicBlock(filter_num, strides))
        for _ in range(1, blocks):  # 其他BasicBlock步长都为1
            res_block.add(BasicBlock(filter_num, strides))
        return res_block


def resnet18():
    return ResNet([2, 2, 2, 2])

