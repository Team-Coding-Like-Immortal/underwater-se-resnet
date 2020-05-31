from vgg import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model

# 打开图片并将图片像素以矩阵的形式保存到列表里
def import_dataset(csv_file):
    datas=[]

    file = pd.read_csv(r'C:/Users/hp/Desktop/VGG/af2020cv-2020-05-09-v5-dev/'+csv_file)
    data = file['FileID']  # 获取名字为flow列的数据
    list = data.values.tolist()  # 将csv文件中flow列中的数据保存到列表中

    for path in list:
        # change_size('af2020cv-2020-05-09-v5-dev/data/'+path+'.jpg')
        datas.append(np.array(Image.open
                              ('af2020cv-2020-05-09-v5-dev/data/'+path+'.jpg', 'r')))

    label = file['SpeciesID']
    labels = label.values.tolist()

    return datas, labels

def main():
    print("正在导入数据")
    # 声明
    training = 'training.csv'
    test = 'annotation.csv'
    train_datas = []
    train_labels = []
    test_datas = []
    test_lables = []

    # 导入数据集（224，224，3，2194）
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

    print('正在训练模型')
    model = VGG(X_train, Y_train)
    model.fit(X_train, Y_train, batch_size=128, epochs=10)
    Y_predict = model.predict(X_train)
    print(Y_predict)
    loss, acc = model.evaluate(X_test, Y_test)
    print(loss, acc)

# 保存模型信息到文件里
    model.save('vgg_model.h5')
'''
# 预测测试集并使用图表将识别的结果显示-----------------------------------------------------------------------------------------------------------出来
with open(test, 'rt') as f:
    test_list = csv.reader(f)
    for test_female_path in test_list:
        open_test(d, test_path[0])
'''
if __name__ == '__main__':
    main()