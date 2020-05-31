import pandas as pd
import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

import os


# 打开图片并将图片像素以矩阵的形式保存到列表里
def import_dataset(csv_file):
    datas=[]

    path_file = os.getcwd() + '/af2020cv-2020-05-09-v5-dev/'
    file = pd.read_csv(path_file + csv_file)
    data = file['FileID']  # 获取名字为flow列的数据
    list = data.values.tolist()  # 将csv文件中flow列中的数据保存到列表中

    for path in list:
        change_size('af2020cv-2020-05-09-v5-dev/data/'+path+'.jpg')
        datas.append(np.array(Image.open
                              ('af2020cv-2020-05-09-v5-dev/data/'+path+'.jpg', 'r')))

    label = file['SpeciesID']
    labels = label.values.tolist()

    return datas, labels


# 将待测试照片大小转化为224*224*3
def change_size(path):
    img = cv2.imread(path)
    width = 224
    height = 224
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim)
    cv2.imwrite(path, resized)

# 模型

def VGG(X):

    model=Sequential()
    #layer_1
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=X.shape[1:],padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',data_format='channels_last',kernel_initializer='uniform',activation='relu'))
    model.add(MaxPooling2D((2,2)))

    #layer_2
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))

    #layer_3
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #layer_4
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))

    #layer_5
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))


    model.add(Flatten())  #拉平
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(20,activation='softmax'))

    model.summary()
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

