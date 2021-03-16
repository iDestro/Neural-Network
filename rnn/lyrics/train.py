import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import time
from utils import predict_rnn


def train(model, preprocess, num_epochs, batch_size, num_steps, lr):
    optimizer = Adam(lr=lr, clipnorm=0.01)
    now = time.time()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    cnt = 0
    for epoch in range(num_epochs):
        l_sum = 0
        n = 0
        data_iter = preprocess.get_data_iter(batch_size=batch_size, num_steps=num_steps)
        for x, y in data_iter:
            with tf.GradientTape(persistent=True) as tape:
                x = tf.one_hot(x, depth=len(preprocess.idx_to_char))
                whole_sequence, _, _ = model(x)
                l = loss(y, whole_sequence)
                grads = tape.gradient(l, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                l_sum += np.array(l).item() * len(y)
                n += len(y)

            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time()-now))
            if n % 50 == 0:
                print(predict_rnn('分开', 20, model, len(preprocess.idx_to_char), preprocess.idx_to_char, preprocess.char_to_idx))







