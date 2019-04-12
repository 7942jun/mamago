import tensorflow as tf

from data.data_loader import DataLoader
from models.seq2seq import Seq2Seq
from trainers.trainer import Trainer
from configs.config import Config

data_loader = DataLoader()
config = Config(data_loader.dic_len)
seq2seq = Seq2Seq(config)
sess = tf.Session()
trainer = Trainer(sess, data_loader, config, seq2seq)

trainer.train()
trainer.test()
