import tensorflow as tf
from tensorflow import keras


class Dense(keras.layers.Layer):
    def __init__(self, units, bias=True, **kwargs):
        """
        :param units: output size
        :param bias: bias item
        :param kwargs:
        """
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.bias = bias

    def call(self, inputs, **kwargs):
        """
        :param inputs: [batch_size, ..., last_dim]
        :param kwargs:
        :return:
        """
        last_dim = inputs.shape[-1]
        # w is the weight matrix of current layer.
        w = tf.Variable(tf.random.uniform(shape=[last_dim, self.units], minval=-0.05, maxval=0.05))
        # sometime bias is not required, set b equal with 0
        if self.bias:
            b = tf.Variable(tf.random.uniform(shape=[self.units], minval=-0.05, maxval=0.05))
        else:
            b = 0
        outputs = tf.matmul(inputs, w) + b
        return outputs


if __name__ == '__main__':
    # generate (20, 10) matrix
    x = tf.random.uniform(shape=[20, 10])
    print(x)
    dense1 = Dense(5, bias=False)
    print(dense1(x))