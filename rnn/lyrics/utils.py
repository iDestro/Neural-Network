import random
import numpy as np
import tensorflow as tf


class Preprocess(object):
    def __init__(self):
        self.idx_to_char = None
        self.char_to_idx = None
        self.corpus_indices = None

    def load_dataset(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            corpus_chars = f.read()

        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', '')
        corpus_chars = corpus_chars[:10000]

        self.idx_to_char = list(set(corpus_chars))
        self.char_to_idx = dict([(char, i) for i, char in enumerate(self.idx_to_char)])
        self.corpus_indices = [self.char_to_idx[char] for char in corpus_chars]

    def get_data_iter(self, batch_size, num_steps, ctx=None):
        # 减1是因为输出的索引是相应输入的索引加1
        num_examples = (len(self.corpus_indices) - 1) // num_steps
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        random.shuffle(example_indices)

        # 返回从pos开始的长为num_steps的序列
        def _data(pos):
            return self.corpus_indices[pos: pos + num_steps]

        for i in range(epoch_size):
            # 每次读取batch_size个随机样本
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            X = [_data(j * num_steps) for j in batch_indices]
            Y = [_data(j * num_steps + 1) for j in batch_indices]
            yield np.array(X, ctx), np.array(Y, ctx)


def predict_rnn(prefix, num_chars, rnn, vocab_size, idx_to_char, char_to_idx):
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = tf.one_hot(output[-1], depth=vocab_size)
        X = tf.reshape(X, [1, 1, -1])
        o, _, _ = rnn(X)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(tf.argmax(o, axis=2)))
    return ''.join([idx_to_char[i] for i in output])


if __name__ == '__main__':
    from model import LSTM
    preprocess = Preprocess()
    preprocess.load_dataset("./data/jaychou_lyrics.txt")
    model = LSTM(len(preprocess.idx_to_char))
    o = predict_rnn('分开', 20, model, len(preprocess.idx_to_char), preprocess.idx_to_char, preprocess.char_to_idx)
    print(o)
