import tensorflow as tf
from utils import Preprocess
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import time
from model import LSTM


def to_onehot(X, size):  # 本函数已保存在d2lzh_tensorflow2包中方便以后使用
    # X shape: (batch), output shape: (batch, n_class)
    return tf.Variable([tf.one_hot(x, size, dtype=tf.float32) for x in X.T])


def train(model, preprocess, num_epochs, batch_size, num_steps, lr):
    data_iter = preprocess.get_data_iter(batch_size=batch_size, num_steps=num_steps)
    optimizer = Adam(lr=lr, clipnorm=1.)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    for epoch in range(num_epochs):
        l_sum = 0
        n = 0
        for x, y in data_iter:
            with tf.GradientTape(persistent=True) as tape:
                x = to_onehot(x, len(preprocess.idx_to_char))
                print(x.shape)
                outputs = model(x)
                print(outputs.shape)
                y = tf.transpose(y)
                print(y.shape)
                l = loss(y, outputs)
                grads = tape.gradient(l, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 因为已经误差取过均值，梯度不用再做平均
                l_sum += np.array(l).item() * len(y)
                n += len(y)

                print('epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, math.exp(l_sum / n), time.time()))


# if __name__ == '__main__':
#     train(200, 1, 5, 0.01)





