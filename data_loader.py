import numpy as np

from config import FLAGS


class DataLoader:
    def __init__(self):
        characters = 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑'
        self.char_list = [c for c in characters]
        self.seq_data = [['word', '단어'], ['wood', '나무'],
                         ['game', '놀이'], ['girl', '소녀'],
                         ['kiss', '키스'], ['love', '사랑']]

    def make_batch(self, seq_data):
        input_batch = []
        output_batch = []
        target_batch = []

        num_dic = {n: i for i, n in enumerate(self.char_list)}

        for seq in seq_data:
            input_data = [num_dic[n] for n in seq[0]]
            output_data = [num_dic[n] for n in ('S' + seq[1])]
            target_data = [num_dic[n] for n in (seq[1] + 'E')]

            input_batch.append(np.eye(FLAGS.input_size)[input_data])
            output_batch.append(np.eye(FLAGS.output_size)[output_data])
            target_batch.append(target_data)

        return input_batch, output_batch, target_batch
