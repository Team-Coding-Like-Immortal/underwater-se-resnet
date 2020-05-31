import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import datetime
import os
from resnet import resnet18
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras_squeeze_excite_network.se_resnet import SEResNet18



devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.random.set_seed(2345)
current_time = datetime.datetime.now().strftime(('%Y-%m-%d_%H-%M-%S'))
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


# 打开图片并将图片像素以矩阵的形式保存到列表里
def import_dataset(csv_file):
    datas = []

    file = pd.read_csv(r'af2020cv-2020-05-09-v5-dev/' + csv_file)
    data = file['FileID']  # 获取名字为flow列的数据
    list = data.values.tolist()  # 将csv文件中flow列中的数据保存到列表中

    for path in tqdm(list):
        change_size('af2020cv-2020-05-09-v5-dev/data/' + path + '.jpg')
        datas.append(np.array(Image.open
                              ('af2020cv-2020-05-09-v5-dev/data/' + path + '.jpg', 'r')))

    datas = np.array(datas)
    label = file['SpeciesID']
    labels = label.values.tolist()
    # print(labels)
    labels = np.array(labels).reshape(len(labels), 1)

    return datas, labels


# 将待测试照片大小转化为32*32*3
def change_size(path):
    img = cv2.imread(path)
    # print(img)
    width = 256
    height = 256
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim)
    cv2.imwrite(path, resized)


def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


print("正在导入数据")
# 声明
training = 'training.csv'
test = 'annotation.csv'

# 导入数据集（32，32，3，... ）
x, y = import_dataset(training)
x_test, y_test = import_dataset(test)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(256)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    # model = resnet18()
    model = SEResNet18()
    model.build(input_shape=(None, 256, 256, 3))
    model.summary()  # 统计网络参数
    optimizer = optimizers.Adam(lr=1e-3)
    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    variables = model.trainable_variables
    for epoch in tqdm(range(1, 6)):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = model(x)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=20)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = model(x)
            prob = tf.nn.softmax(out, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print('acc: %d' % acc)
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(acc), step=epoch)


if __name__ == '__main__':
    main()
