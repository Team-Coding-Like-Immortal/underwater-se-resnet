import os
import sys
import time
import pickle
import random
import numpy as np

img_size = 32
img_channels = 3

# How to load data
    #train_X, train_lab, test_X, test_lab = get_data()
    #train_X, test_X = normalize(train_X, test_X)


def unpickle10(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data_X = dict[b'data']
    labels = dict[b'labels']
    print("Loading %s : %d." % (file, len(data_X)))
    return data_X, labels

def unpickle100(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data_X = dict[b'data']
    labels = dict[b'fine_labels']
    print("Loading %s : %d." % (file, len(data_X)))
    return data_X, labels


def load_data10(files, data_dir, label_count):
    global img_size, img_channels
    data, labels = unpickle10(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n =  unpickle10(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    #labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    # channels_last
    data = data.reshape([-1, img_channels, img_size, img_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def load_data100(files, data_dir, label_count):
    global img_size, img_channels
    data, labels = unpickle100(data_dir + '/' + files[0])
    #labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    # channels_last
    data = data.reshape([-1, img_channels, img_size, img_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def get_data(dataset):
    img_dim = img_size * img_size * img_channels
    if dataset == 100:
        n_labels = 100
        #data_dir = './Datasets/cifar-100-python'
        data_dir = '../cifar-100-python'

        train_data, train_labels = load_data100(['train'], data_dir, n_labels)
        test_data, test_labels = load_data100(['test'], data_dir, n_labels)

    else:
        n_labels = 10
        data_dir = './Datasets/cifar-10-batches-py'

        train_files = ['data_batch_%d' % d for d in range(1, 6)]
        train_data, train_labels = load_data10(train_files, data_dir, n_labels)
        test_data, test_labels = load_data10(['test_batch'], data_dir, n_labels)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))

    # Shuffle
    ind = np.random.permutation(len(train_data))
    train_data = train_data[ind]
    train_labels = np.array(train_labels)
    train_labels = train_labels[ind]

    return train_data, train_labels, test_data, test_labels

def normalize(train_X, test_X):
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    mean_X = np.mean(train_X / np.max(train_X))
    std_X = np.std(train_X / np.max(train_X))

    train_norm = train_X / np.max(train_X)
    train_norm = (train_norm - mean_X) / std_X

    test_norm = test_X / np.max(test_X)
    test_norm = (test_norm - mean_X) / std_X

    return train_norm, test_norm


# =============================================================================== #
# Functions for data augmentation (not really needed for a comparison study)
# =============================================================================== #


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                        mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                        nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch