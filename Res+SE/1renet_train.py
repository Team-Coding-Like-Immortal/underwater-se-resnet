import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import datetime
import os
from resnet import resnet18
devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.random.set_seed(2345)
current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
log_dir = 'logs/'+current_time
#summary_writer = tf.summary.create_file_writer(log_dir)



def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()
print(y,x)
y = tf.squeeze(y, axis=1)
print(y,x)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(256)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

'''
def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary() # 统计网络参数
    optimizer = optimizers.Adam(lr=1e-3)
    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    variables = model.trainable_variables
    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = model(x)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)
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
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(acc), step=epoch)


if __name__ == '__main__':
     main()
'''