import os, glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import datetime
# from resnet import resnet18
from keras_squeeze_excite_network.se_resnet import SEResNet18
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import random, csv
import smtplib
from email.mime.text import MIMEText
from email.header import Header

devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.random.set_seed(2345)
current_time = datetime.datetime.now().strftime(('%Y-%m-%d_%H:%M:%S'))
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
os.makedirs('weights', exist_ok=True)

# 打开图片并将图片像素以矩阵的形式保存到列表里
# def import_dataset(csv_file):
#     datas=[]

#     file = pd.read_csv(r'af2020cv-2020-05-09-v5-dev/'+csv_file)
#     data = file['FileID']  # 获取名字为flow列的数据
#     list = data.values.tolist()  # 将csv文件中flow列中的数据保存到列表中

#     for path in tqdm(list):
#         # change_size('af2020cv-2020-05-09-v5-dev/data/'+path+'.jpg')
#         datas.append(np.array(Image.open('af2020cv-2020-05-09-v5-dev/data/'+ path +'.jpg', 'r')))

#     datas=np.array(datas)
#     label = file['SpeciesID']
#     labels = label.values.tolist()
#     labels =np.array(labels).reshape(len(labels),1)

#     return datas, labels


# # 将待测试照片大小转化为32*32*3
# def change_size(path):
#     img = cv2.imread(path)
#     # print(img)
#     width = 224
#     height = 224
#     dim = (width, height)

#     # resize image
#     resized = cv2.resize(img, dim)
#     cv2.imwrite(path, resized)


# def preprocess(x, y):
#     # [0~1]
#     x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
#     y = tf.cast(y, dtype=tf.int32)
#     return x, y


assert tf.__version__.startswith('2.')


class Reminder:
    def __init__(self, qq=None, register=None):
        """
        :param qq: 发送的qq账号
        :param register: qq邮箱授权吧
        """
        self.qq = qq
        self.register = register
        self.server = smtplib.SMTP_SSL("smtp.qq.com", 465)

    def send(self, title, detail):
        """
        send message
        :param title: the title of the message
        :param detail: the detail of the message
        """
        sender = self.qq
        receivers = self.qq
        message = MIMEText(detail, 'plain', 'utf-8')
        message['Subject'] = Header(title, 'utf-8')
        message['From'] = sender
        message['To'] = receivers
        try:
            self.server = smtplib.SMTP_SSL("smtp.qq.com", 465)
            self.server.login(sender, self.register)
            self.server.sendmail(sender, receivers, message.as_string())
            self.server.quit()
        except smtplib.SMTPException as e:
            print(e)

    def _register(self):
        self.qq = '434596665@qq.com'
        self.register = 'qbcomikcojwubgca'


def load_csv(root, filename):
    images, labels = [], []
    # read from csv file
    with open(os.path.join(root, filename), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in tqdm(reader):
            img, label = row
            img = os.path.join(root, 'data', img) + '.jpg'
            label = int(label)
            images.append(img)
            labels.append(label)
    assert len(images) == len(labels)
    return images, labels


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
    # resize the image,you can change the value in the another net
    x = tf.image.resize(x, [224, 224])
    # turn around images
    x = tf.image.random_crop(x, [224, 224, 3])
    # # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return x, y


print("正在导入数据")
# 声明
training = 'training.csv'
test = 'annotation.csv'

# 导入数据集（32，32，3，... ）
# x, y = load_s(training)
x, y = load_csv('af2020cv-2020-05-09-v5-dev', 'training.csv')
x_test, y_test = load_csv('af2020cv-2020-05-09-v5-dev', 'annotation.csv')
# x_test, y_test = import_dataset(test)
# y = tf.squeeze(y, axis=1)
# y_test = tf.squeeze(y_test, axis=1)
# print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(32)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main(epochs, lr, ckpt_interval, ckpt_path, weights_path):
    reminder = Reminder()
    reminder._register()
    count = 0
    temp = 0
    content_list = []
    # 模型训练
    print('开始训练')
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    # model = resnet18()
    model = SEResNet18()
    # model = ResNet50(include_top=True, squeeze=True, squeeze_type='pre', classes=20)
    model.build(input_shape=(None, 224, 224, 3))
    # model.summary() # 统计网络参数
    # optimizer = optimizers.Adam(lr=1e-3)
    optimizer_init = optimizers.SGD(lr=lr)
    optimizer_sgd1 = optimizers.SGD(learning_rate=lr / 10)
    optimizer_sgd2 = optimizers.SGD(learning_rate=lr / 100)
    optimizer_sgd3 = optimizers.SGD(learning_rate=lr / 1000)
    optimizer = optimizer_init
    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    model.load_weights(weights_path)
    variables = model.trainable_variables
    for epoch in tqdm(range(1, epochs + 1)):
        optimizer = optimizer_init
        if epoch == 20:
            optimizer = optimizer_sgd1
        if epoch == 50:
            optimizer = optimizer_sgd2
        if epoch == 80:
            optimizer = optimizer_sgd3
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = model(x)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=20)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True)
                loss = tf.reduce_mean(loss)
            # print('loss:', loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % 100 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('loss', float(loss), step=step)

        # print('loss: %g, acc: %g' % (loss, acc))

        # tf.keras.models.save_model(
        #     model, ckpt_path + 'tf_ckpt_%d.h5' % epoch, include_optimizer=True,
        #     save_format='tf', signatures=None, options=None
        # )
        try:
            if epoch % ckpt_interval == 0:
                print('保存模型中.....')
                model.save_weights(ckpt_path + 'tf_ckpt_%d.h5' % epoch)
                print('保存成功')
        except KeyboardInterrupt:
            print('训练被中断...正在保存模型')
            model.save_weights(ckpt_path + 'tf_ckpt_%d.h5' % epoch)
            print('保存成功')

        total_num = 0
        total_correct = 0
        # 模型测试
        print('\n目前是 %d 个epoch结束之后的测试,总共 %d' % (epoch, epochs))
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
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(acc), step=epoch)
            print('loss: %f, acc: %f' % (loss, acc))
        if epoch % 10 == 0:
            content_list.append('Epoch %d, Loss: %f, acc: %f\n' % (epoch, loss, acc))
        #
        if temp > acc:
            count += 1
            if count > 5:
                break
        else:
            count = 0
        temp = acc

    reminder.send('SE-ResNet Training', str(content_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="adjust the learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="where you save the checkpoint")
    parser.add_argument("--weight_dir", type=str, default="weights/", help="where you save the checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="how many epochs after saving model")
    parser.add_argument("--evaluate_interval", type=int, default=1, help="how many epochs per evaluate model")
    parser.add_argument("--train_path", type=str, default="training.csv", help="where the train image csv file")
    parser.add_argument("--test_path", type=str, default="annotation.csv", help="where the test image csv file")

    os.makedirs("checkpoints", exist_ok=True)

    opt = parser.parse_args()
    print(opt)

    main(opt.epoch, opt.learning_rate, opt.checkpoint_interval, opt.checkpoint_dir, opt.weight_dir)