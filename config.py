import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('train_dir', './models', 'Trained model save folder')
flags.DEFINE_string('checkpoint_name', 'mamago.checkpoint', 'Name of checkpoint')
flags.DEFINE_string('log_dir', './logs', '로그를 저장할 폴더')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learing rate.')
flags.DEFINE_integer('hidden', 128, 'Number of units in hidden layer')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainers')

dic_len = 41
flags.DEFINE_integer('input_size', dic_len, 'Size of input')
flags.DEFINE_integer('output_size', dic_len, 'Size of output')

flags.DEFINE_boolean("train", False, "학습을 진행합니다.")
flags.DEFINE_boolean("test", True, "테스트를 진행합니다.")

FLAGS = flags.FLAGS