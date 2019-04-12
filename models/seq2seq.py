import tensorflow as tf


class Seq2Seq:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.enc_input = tf.placeholder(
            tf.float32, [None, None, self.config.n_input])
        self.dec_input = tf.placeholder(
            tf.float32, [None, None, self.config.n_input])
        self.targets = tf.placeholder(tf.int64, [None, None])

        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.n_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(
                enc_cell, input_keep_prob=0.5)

            outputs, enc_states = tf.nn.dynamic_rnn(
                enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.n_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(
                dec_cell, output_keep_prob=0.5)

            outputs, _ = tf.nn.dynamic_rnn(dec_cell, self.dec_input,
                                                    initial_state=enc_states, dtype=tf.float32)

        self.model = tf.layers.dense(outputs, self.config.n_class, activation=None)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.targets)
        self.cost = tf.reduce_mean(entropy)

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost)
