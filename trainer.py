import tensorflow as tf
import os

from config import FLAGS
from seq2seq import Seq2Seq
from data_loader import DataLoader


def train():
    model = Seq2Seq()
    data_loader = DataLoader()

    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
        if checkpoint:
            print('다음 파일에서 모델을 읽는 중입니다... ', checkpoint)
            model.saver.restore(sess, checkpoint)
        else:
            print('새로운 모델을 생성하는 중입니다...')
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        input_batch, output_batch, target_batch = data_loader.make_batch(data_loader.seq_data)

        for step in range(FLAGS.max_steps):
            _, loss = model.train(
                sess, input_batch, output_batch, target_batch)
            
            model.write_logs(sess, writer, input_batch, output_batch, target_batch)

            print('Step', '%04d' % (step + 1), 'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.checkpoint_name)
        model.saver.save(sess, checkpoint_path)
    print('최적화 완료!')


def test():
    model = Seq2Seq()
    data_loader = DataLoader()

    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
        if checkpoint:
            print('다음 파일에서 모델을 읽는 중입니다... ', checkpoint)
            model.saver.restore(sess, checkpoint)

            while True:
                word = input('Enter an english word: ')
                seq_data = [word, 'P' * len(word)]

                input_batch, output_batch, target_batch = data_loader.make_batch([
                    seq_data])
                prediction = tf.argmax(model.logits, 2)
                result = sess.run(prediction, feed_dict={model.enc_input: input_batch,
                                                        model.dec_input: output_batch,
                                                        model.targets: target_batch})

                decoded = [data_loader.char_list[i] for i in result[0]]
                try:
                    end = decoded.index('E')
                    translated = ''.join(decoded[:end])
                except:
                    translated = ''.join(decoded)

                print(word, ' -> ', translated)
        else:
            print('학습된 데이터가 없습니다.')

def main(_):
    if FLAGS.train:
        train()
    elif FLAGS.test:
        test()

if __name__ == '__main__':
    tf.app.run()
