import tensorflow as tf

from config import FLAGS


class Seq2Seq:
    logits: None
    cost: None
    optimizer: None

    def __init__(self):
        self.enc_input = tf.placeholder(
            tf.float32, [None, None, FLAGS.input_size])
        self.dec_input = tf.placeholder(
            tf.float32, [None, None, FLAGS.input_size])
        self.targets = tf.placeholder(tf.int64, [None, None])

        self.build_model()

        self.saver = tf.train.Saver()

    def _cell(self, output_keep_prob=0.5):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_ops(self, outputs, targets):
        logits = tf.layers.dense(
            outputs, FLAGS.output_size, activation=None)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets))
        optimizer = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(cost)

        tf.summary.scalar('cost', cost)

        return logits, cost, optimizer

    def build_model(self):
        enc_cell = self._cell()
        dec_cell = self._cell()

        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(
                enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            outputs, _ = tf.nn.dynamic_rnn(dec_cell, self.dec_input,
                                           initial_state=enc_states, dtype=tf.float32)

        self.logits, self.cost, self.optimizer = self.build_ops(
            outputs, self.targets)

    def train(self, sess, enc_input, dec_input, targets):
        return sess.run([self.optimizer, self.cost], feed_dict={self.enc_input: enc_input,
                                                                self.dec_input: dec_input,
                                                                self.targets: targets})

    def write_logs(self, sess, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()
        summary = sess.run(merged, feed_dict={self.enc_input: enc_input,
                                              self.dec_input: dec_input,
                                              self.targets: targets})
        writer.add_summary(summary)
