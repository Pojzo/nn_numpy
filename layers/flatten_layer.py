from .base_layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        super().__init__(activation=None)

    def forward(self, inpt):
        if len(inpt.shape) == 3:
            batch_size = inpt.shape[0]

        return np.reshape(inpt, (batch_size, inpt.shape[1] * inpt.shape[2]))

    def backward(self, d_next):
        return d_next, 0, 0
