import tensorflow as tf


class Trainer:
    def __init__(self, sess, data_loader, config, model):
        self.sess = sess
        self.data_loader = data_loader
        self.config = config
        self.model = model

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        input_batch, output_batch, target_batch = self.data_loader.make_batch(
            self.data_loader.seq_data)

        for epoch in range(self.config.total_epoch):
            _, loss = self.sess.run(
                [self.model.optimizer, self.model.cost], feed_dict={self.model.enc_input: input_batch,
                                                                    self.model.dec_input: output_batch,
                                                                    self.model.targets: target_batch})

            print('Epoch', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))
        print('최적화 완료!')

    def test(self):
        while True:
            word = input('Enter an english word: ')
            seq_data = [word, 'P' * len(word)]

            input_batch, output_batch, target_batch = self.data_loader.make_batch([
                seq_data])
            prediction = tf.argmax(self.model.model, 2)
            result = self.sess.run(prediction, feed_dict={self.model.enc_input: input_batch,
                                                    self.model.dec_input: output_batch,
                                                    self.model.targets: target_batch})

            decoded = [self.data_loader.char_list[i] for i in result[0]]
            end = decoded.index('E')
            translated = ''.join(decoded[:end])

            print(word, ' -> ', translated)
