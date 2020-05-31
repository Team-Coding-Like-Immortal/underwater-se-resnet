"""
Load PoKemon dataset
"""
import os, glob
import random, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
assert tf.__version__.startswith('2.')





def load_csv(root, filename):
    images, labels = [], []
    # read from csv file
    with open(os.path.join(root, filename), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            img = os.path.join(root,'data',img) + '.jpg'
            label = int(label)
            images.append(img)
            labels.append(label)
    assert len(images) == len(labels)
    return images, labels

#
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean) / std
    return x


def preprocess(x, y):
    """
    preprocess the data
    :param x: the path of the images
    :param y: labels
    """
    # data augmentation, 0~255
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    #     # resize the image,you can change the value in the another net
    #     x = tf.image.resize(x, [224, 224])
    #     # turn around images
    #     x = tf.image.random_crop(x, [224, 224, 3])
    #     # # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return x,y


if __name__ == '__main__':
    image_train, lab_train = load_csv('af2020cv-2020-05-09-v5-dev', 'training.csv')
    print(image_train)
    # train_db = tf.data.Dataset.from_tensor_slices((image_train, lab_train))
    # train_db = train_db.shuffle(1000).map(preprocess).batch(32)
    # val_db = tf.data.Dataset.from_tensor_slices((image_val, lab_val))
    # val_db = val_db.map(preprocess).batch(32)
    # test_db = tf.data.Dataset.from_tensor_slices((image_test, lab_test))
    # test_db = test_db.map(preprocess).batch(32)
    # print(train_db)
    # print(val_db)
    # print(test_db)