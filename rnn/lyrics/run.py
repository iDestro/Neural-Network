from utils import Preprocess
from train import train
from model import LSTM
from config import Config

config = Config()
preprocess = Preprocess()
preprocess.load_dataset(config.path)
model = LSTM(len(preprocess.idx_to_char))

train(model=model,
      preprocess=preprocess,
      batch_size=config.batch_size,
      num_epochs=config.num_epochs,
      num_steps=config.num_steps,
      lr=config.lr
      )
