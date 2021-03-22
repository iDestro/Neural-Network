from tensorflow.keras.layers import Layer
import tensorflow as tf


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.h_t = self.add_weight(
            shape=[input_dim, 1],
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='h_t'
        )
        self.attention_weight = self.add_weight(
            shape=[input_dim, input_dim],
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='attention_weight'
        )

    def call(self, inputs, **kwargs):
        """
        :param inputs: 3D (batch_size, time_steps, input_dim)
        :param kwargs:
        :return:
        """
        input_dim = tf.shape(inputs)[-1]
        time_steps = tf.shape(inputs)[-2]
        score = tf.matmul(tf.matmul(tf.reshape(inputs, [-1, input_dim]), self.attention_weight), self.h_t)
        score = tf.reshape(score, [-1, time_steps, 1])
        alpha = tf.nn.softmax(score, axis=1)
        c_t = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alpha)
        return c_t


if __name__ == '__main__':
    att = Attention()
    a = tf.random.uniform([10, 3, 16])
    print(att(a))