import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.size_layers = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x, y in zip(sizes[0:self.n_layers-1], sizes[1:self.n_layers])]

    
