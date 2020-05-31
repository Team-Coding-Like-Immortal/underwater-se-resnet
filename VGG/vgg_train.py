from vgg import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from torch.autograd import *
import argparse


def main(epoch, batch_szie, training, test, evaluate_interval):
    print("开始导入数据")
    # 声明
    # training = 'training.csv'
    # test = 'annotation.csv'
    train_datas = []
    train_labels = []
    test_datas = []
    test_lables = []

    # 导入数据集（32，32，3，2194）
    train_datas, train_labels = import_dataset(training)
    test_datas, test_lables = import_dataset(test)
    # 对数据进行处理
    # Convert and pre-processing
    train_datas = np.array(train_datas, dtype="float") / 255.0
    train_labels = np.array(train_labels)
    test_datas = np.array(test_datas, dtype="float") / 255.0
    test_lables = np.array(test_lables)
    X_train = train_datas.reshape(-1, 224, 224, 3)
    X_test = test_datas.reshape(-1, 224, 224, 3)
    Y_train = to_categorical(train_labels, num_classes=20)
    Y_test = to_categorical(test_lables, num_classes=20)
    print('导入完毕')

    try:
        print('开始训练模型')
        model = VGG(X_train)
        model.load_weights('vgg_model.h5')
        model.fit(X_train, Y_train, batch_size=batch_szie, epochs=epoch)
        print('开始预测')
        Y_predict = model.predict(X_train)
        print('开始评估')
        loss, acc = model.evaluate(X_test, Y_test)
        print('loss: %g    acc: %g' % (loss, acc))
        # 保存模型信息到文件里
        name = str(epoch) + 'vgg_model.h5'
        model.save(name)
    except KeyboardInterrupt:
        # 保存模型信息到文件里
        model.save('vgg_model_interrupt.h5')

"""
# 预测测试集并使用图表将识别的结果显示-----------------------------------------------------------------------------------------------------------出来
with open(test, 'rt') as f:
    test_list = csv.reader(f)
    for test_female_path in test_list:
        open_test(d, test_path[0])
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch size")
    parser.add_argument("--evaluate_interval", type=int, default=1, help="how many epochs per evaluate model")
    parser.add_argument("--train_path", type=str, default="training.csv", help="where the train image csv file")
    parser.add_argument("--test_path", type=str, default="annotation.csv", help="where the test image csv file")
    
    opt = parser.parse_args()
    print(opt)

    main(opt.epoch, opt.batch_size, opt.train_path, opt.test_path, opt.evaluate_interval)
