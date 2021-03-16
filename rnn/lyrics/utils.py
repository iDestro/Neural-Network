import random
import numpy as np


class Preprocess(object):
    def __init__(self):
        self.idx_to_char = None
        self.char_to_idx = None
        self.corpus_indices = None

    def load_dataset(self, path):
        with open(path, 'r') as f:
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