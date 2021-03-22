from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class Embedding(Layer):
    def __init__(self, vocab_size, model_dim, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.model_dim = model_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.model_dim),
            initializer='glorot_uniform',
            name='embeddings'
        )
        super(Embedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        # gather 将inputs的元素以inputs对应的值为索引替换为embedding位置的向量
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self.model_dim ** 0.5
        return embeddings

    def compute_output_shape(self, input_shape):
        # concatenate
        return input_shape + (self.model_dim, )


class PositionEncoding(Layer):
    def __init__(self, model_dim, **kwargs):
        self.model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self.model_dim))
        for pos in range(seq_length):
            for i in range(self.model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i % 2) / self.model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ScaledDotProductAttention(Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.masking = masking
        self.future = future
        self.dropout_rate = dropout_rate
        self.masking_num = -2 ** 32 + 1

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self.masking_num
        return outputs

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self.masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs, **kwargs):
        if self.masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]"
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]"
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':
            queries = K.cast(queries, 'float32')

        if K.dtype(keys) != 'float32':
            keys = K.cast(keys, 'float32')

        if K.dtype(values) != 'float32':
            values = K.cast(values, 'float32')

        # (batch_size*n_heads, max_len, head_dim)
        # (batch_size*n_heads, head_dim, max_len)
        # (batch_size*n_heads, max_len, max_len)
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale

        if self.masking:
            scaled_matmul = self.mask(scaled_matmul, masks)

        if self.future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        # TODO: 这里的dropout是做什么的
        # Dropout
        out = K.dropout(softmax_out, self.dropout_rate)
        # TODO: batch_dot的实际意义
        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):
    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.masking = masking
        self.future = future
        self.trainable = trainable

    # TODO: q, k, v 的大小之间有什么关系?
    def build(self, input_shape):
        self.weights_queries = self.add_weight(
            shape=[input_shape[0][-1], self.n_heads * self.head_dim],
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_queries'
        )
        self.weights_keys = self.add_weight(
            shape=[input_shape[1][-1], self.n_heads * self.head_dim],
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_keys'
        )
        self.weights_values = self.add_weight(
            shape=[input_shape[2][-1], self.n_heads * self.head_dim],
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_values'
        )

    def call(self, inputs, **kwargs):
        if self.masking:
            assert len(inputs) == 4, 'inputs should be set [queries, keys, values, masks].'
            queries, keys, values, masks = inputs

        else:
            assert len(inputs) == 3, 'inputs should be set [queries, keys, values].'
            queries, keys, values = inputs

        # (batch_size, max_len, model_dim) * (model_dim, n_heads, head_dim)
        # = (batch_size, max_len, n_heads, head_dim)
        queries_linear = K.dot(queries, self.weights_queries)
        keys_linear = K.dot(keys, self.weights_keys)
        values_linear = K.dot(values, self.weights_values)

        # TODO: axis=2指的是？
        # 分割后的单个矩阵的shape为：(batch_size, max_len, 1,  head_dim)， 一共有n_heads个
        # (batch_size*n_heads, max_len, 1, head_dim)
        queries_multi_heads = tf.concat(tf.split(queries_linear, self.n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self.n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self.n_heads, axis=2), axis=0)

        if self.masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(masking=self.masking, future=self.future, dropout_rate=self.dropout_rate)
        att_out = attention(att_inputs)
        outputs = tf.concat(tf.split(att_out, self.n_heads, axis=0), axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':
    import os
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.utils import to_categorical

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    vocab_size = 5000
    max_len = 256
    model_dim = 512
    batch_size = 128
    epochs = 10

    print("Data downloading and pre-processing ... ")
    # 返回的是个列表，列表中的元素是向量（索引编号向量）
    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    # 将0即pad上的位置用mask标记好
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    # 转换为独热编码
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Model building ... ')
    inputs = Input(shape=(max_len,), name="inputs")
    masks = Input(shape=(max_len,), name='masks')
    # 随机初始化词向量
    embeddings = Embedding(vocab_size, model_dim)(inputs)
    # 采用PositionEncoding对词向量进行编码
    encodings = PositionEncoding(model_dim)(embeddings)
    # 直接将embeddings与encodings相加, 得到编码
    encodings = Add()([embeddings, encodings])
    x = MultiHeadAttention(8, 64)([encodings, encodings, encodings, masks])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs, masks], outputs=outputs)
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model Training ... ")
    es = EarlyStopping(patience=5)
    model.fit([x_train, x_train_masks], y_train,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])