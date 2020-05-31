import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf		
import datetime

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras.backend as K
from Squeeze_and_Excite import Squeeze_and_Excite 
from ResNet50 import *
from LoadDataset import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

#%load_ext tensorboard
#!rm -rf ./logs/

epoch = 10
batch_size = 128
learning_rate = 0.01
dataset = 100

"""
class layerModel(Model):
    def __init__(self, inp_shape):
        super(layerModel, self).__init__()
        self.squeeze = Squeeze_and_Excite(inp_shape, 3) # ratio = 3 
        self.reshape = Flatten()  #Reshape((3072, 1))
        self.h = Dense(100, activation = 'relu', kernel_initializer = 'glorot_uniform', input_shape=(inp_shape, ))
        self.p = Dense(10, activation ='softmax')

    def call(self, x):
        y = self.squeeze(x)
        y = self.reshape(y)
        y = self.h(y)
        y = self.p(y)
        return y

#model = layerModel(3)
"""

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='test_1_accuracy') 
test_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='test_5_accuracy') 

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training = True)
        print(predictions.get_shape())
        loss =loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
        
@tf.function
def test_step(images, labels):
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training = False)
    t_loss =loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_top1(labels, predictions)
    test_top5(labels, predictions)


if __name__ == "__main__":

    save_model_dir = ".\checkpoints"

    print("loading data...")
    train_X, train_lab, test_X, test_lab = get_data(dataset)
    print("normalizing data...")
    train_X, test_X = normalize(train_X, test_X)
    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_lab)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_lab)).batch(batch_size)

    #model = ResNet50(include_top=True, weights=None, squeeze=False, squeeze_type='Normal')#, input_tensor = tf.data.Dataset.from_tensor_slices((train_X)))
    #input_tensor = tf.placeholder(tf.float32, shape = [None, train_X.shape[0], train_X.shape[1], train_X.shape[2]])
    model = ResNet50(include_top=True, squeeze=False, squeeze_type='normal', classes=dataset) #pre, identity, normal

    """
    features, label = iter(train_dataset).next()
    print("example features:", features[0])
    print("example label:", label[0])
    """

    checkpoint_dir = os.path.join(save_model_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for i in range(epoch):
        for images, labels in train_data:
            print(tf.shape(images))
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i+1)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=i+1)

            
        for images, labels in test_data:
            test_step(images, labels)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=i+1)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=i+1)

        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(i+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
						test_loss.result(),
                        test_accuracy.result()*100))

        template = 'Top1 Error: {}, Top5 Error: {}'
        print(template.format((1 - test_top1.result())*100,
						(1 - test_top5.result())*100))

        save_path = manager.save()

        train_losses.append(train_loss.result())
        train_accs.append(train_accuracy.result())
        test_losses.append(test_loss.result())
        test_accs.append(test_accuracy.result())

        # Reinicia las metricas para el siguiente epoch.
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


    t = np.linspace(1, epoch, num=epoch)
    plot1 = plt.figure(1)
    plt.plot(t, train_losses, 'b')
    plt.plot(t, test_losses, 'r')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training loss evolution')
    plt.savefig('.\Result_Pics\loss_train')
    plot2 = plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(t, train_accs, 'b')
    plt.plot(t, test_accs, 'r')
    plt.title('Training accuracy evolution')
    plt.savefig('.\\Result_Pics\\acc_train')
    plt.show()

    #run in the command line: tensorboard --logdir [log dir]

    a = 1