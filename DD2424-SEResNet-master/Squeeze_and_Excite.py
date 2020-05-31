from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Reshape, Multiply
from tensorflow.keras import Model

class Squeeze_and_Excite(Model):
     def __init__(self, s, r = 2):
        super(Squeeze_and_Excite, self).__init__()
        self.squeeze = GlobalAveragePooling2D(data_format = 'channels_last')
        self.dense_rel = Dense(s//r, activation = 'relu', kernel_initializer = 'he_normal')
        self.dense_sig = Dense(s, activation = 'sigmoid', kernel_initializer = 'he_normal')
       
     def call(self, x):
        y = self.squeeze(x)
        y = self.dense_rel(y)
        y = self.dense_sig(y)
        return Multiply()([x, y])