class Parameter():
    def __init__(self):
        self.batch_size = 20
        self.is_mlp = True
        self.block_size = 10
        self.feature_length = 200
        self.speaker_num = 107
        self.learning_rate = 0.001
        self.lstm_size = 50
        self.num_layer = 3
        self.block_ms = 50.0
        self.correct = None
        self.total = None
        self.predict = None
        self.max_gradient_norm = 1
parameter = Parameter()