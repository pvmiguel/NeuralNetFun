"""
Network.py
-----------

"""

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        #DONE
        self.n_layers = len(sizes)
        self.size_layers = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[0:self.n_layers-1], sizes[1:self.n_layers])]

    def feed(self, state):
        """Function that returns the output of the neural network for a given input state"""
        #DONE
        for b, w in zip(self.biases, self.weights):
            state = sigmoid(z(w, state, b))
        return state

    def backprop(self, state, y):
        #DONE
        partialC_b = [np.zeros(b.shape) for b in self.biases]
        partialC_w = [np.zeros(w.shape) for w in self.weights]
        activations = [state]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z_l = z(w, state, b)
            state = sigmoid(z_l)
            zs.append(z_l)
            activations.append(state)
        delta_l = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])
        partialC_b[-1] = delta_l
        partialC_w[-1] = np.dot(delta_l, activations[-2].transpose())
        for n in [2, self.n_layers]
            delta_l = np.dot(self.weights[-n+1].transpose(), delta_l) * sigmoid_prime(zs[-n])
            partialC_b[-n] = delta_l
            partialC_w[-n] = np.dot(delta_l, activations[-n-1].transpose())
        return [partialC_b, partialC_w]

    def SGD(self, training_data, generations, batch_size, eta, test_data=None):
        """Function that performs a sto"""
        return NULL

    def cost_prime(self, a_out, y):
        #DONE
        c_prime = a_out - y;
        return c_prime

def sigmoid(z):
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig

def sigmoid_prime(z):
    sig_p = np.exp(-z) / pow(exp(-z) + 1, 2)
    return sig_p

def z(w, state, b):
    #DONE
    z = np.dot(w, state) + b
    return z
