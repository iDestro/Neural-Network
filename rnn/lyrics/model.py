import tensorflow as tf
from tensorflow import keras


class LSTM(keras.Model):

    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        # self.batch_size = batch_size
        self.lstm = keras.layers.LSTM(units=units)

    # def get_params(self):
    #     def _one(shape):
    #         return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32))
    #
    #     def _three():
    #         return (_one((self.input_dim, self.hidden)),
    #                 _one((self.input_dim, self.hidden_dim)),
    #                 tf.Variable(tf.zeros(self.hidden_dim), dtype=tf.float32))
    #
    #     W_xi, W_hi, b_i = _three()  # 输入门参数
    #     W_xf, W_hf, b_f = _three()  # 遗忘门参数
    #     W_xo, W_ho, b_o = _three()  # 输出门参数
    #     W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
    #
    #     # 输出层参数
    #     W_hq = _one((self.hidden_dim, self.output_dim))
    #     b_q = tf.Variable(tf.zeros(self.output_dim), dtype=tf.float32)
    #     H, C = tf.zeros(shape=(self.batch_size, self.hidden_dim)), tf.zeros(shape=(self.batch_size, self.hidden_dim))
    #     return [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q, H, C]

    def call(self, inputs, training=None, mask=None):
        # [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q, H, C] = self.get_params()
        # outputs = []
        # for X in inputs:
        #     X = tf.reshape(X, [-1, W_xi.shape[0]])
        #     I = tf.sigmoid(tf.matmul(X, W_xi) + tf.matmul(H, W_hi) + b_i)
        #     F = tf.sigmoid(tf.matmul(X, W_xf) + tf.matmul(H, W_hf) + b_f)
        #     O = tf.sigmoid(tf.matmul(X, W_xo) + tf.matmul(H, W_ho) + b_o)
        #     C_tilda = tf.tanh(tf.matmul(X, W_xc) + tf.matmul(H, W_hc) + b_c)
        #     C = F * C + I * C_tilda
        #     H = O * tf.tanh(C)
        #     Y = tf.matmul(H, W_hq) + b_q
        #     outputs.append(Y)
        # return outputs, (H, C)
        return self.lstm(inputs)


def to_onehot(X, size):  # 本函数已保存在d2lzh_tensorflow2包中方便以后使用
    # X shape: (batch), output shape: (batch, n_class)
    return tf.Variable([tf.one_hot(x, size,dtype=tf.float32) for x in X.T])


if __name__ == '__main__':
    model = LSTM()
    from utils import Preprocess
    pre = Preprocess()
    pre.load_dataset('./jaychou_lyrics.txt')

    iter = pre.get_data_iter(batch_size=3, num_steps=5)

    for x, y in iter:
        x = to_onehot(x, len(pre.idx_to_char))
        print(model(x))
        break


