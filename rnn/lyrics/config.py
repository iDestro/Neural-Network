class Config(object):
    def __init__(self):
        self.path = './data/jaychou_lyrics.txt'
        self.batch_size = 5
        self.num_steps = 20
        self.num_epochs = 200
        self.lr = 0.01
