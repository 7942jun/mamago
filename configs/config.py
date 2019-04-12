class Config:
    def __init__(self, dic_len):
        self.learning_rate = 0.01
        self.n_hidden = 128
        self.total_epoch = 100
        self.n_class = self.n_input = dic_len
        