import numpy as np

from config import FLAGS


class DataLoader:
    def __init__(self):
        self.seq_data, characters = self.load_data(FLAGS.data_path)
        self.char_list = [c for c in characters]

    def load_data(self, data_path):
        seq_data = []
        characters = 'SEPabcdefghijklmnopqrstuvwxyz'

        with open(data_path, 'r', encoding='utf-8') as data:
            for word_set in data:
                seq = []
                word_en = word_set.split()[0]
                word_ko = word_set.split()[1]

                seq.append(word_en)
                seq.append(word_ko)

                characters = self.add_new_char(word_ko, characters)
                seq_data.append(seq)
        return seq_data, characters

    def add_new_char(self, word, characters):
        for char in word:
            if char not in characters:
                characters += char
        return characters

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
