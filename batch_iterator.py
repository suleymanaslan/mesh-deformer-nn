import numpy as np


def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


class BatchIterator:
    def __init__(self, training_x, training_y, batch_size):
        self.training_x = training_x
        self.training_y = training_y
        self.batch_size = batch_size
        self.size = self.training_x.shape[0]
        self.epochs = 0
        self.cursor = 0
        self.shuffle()

    def shuffle(self):
        shuffle_arrays((self.training_x, self.training_y))
        self.cursor = 0

    def next_batch(self):
        batch_x = self.training_x[self.cursor:self.cursor + self.batch_size]
        batch_y = self.training_y[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        if self.cursor + self.batch_size - 1 >= self.size:
            self.epochs += 1
            self.shuffle()
        return batch_x, batch_y
