from .base_layer import Layer

import numpy as np

class Normalize(Layer):
    def __init__(self):
        super().__init__(activation=None)

    def forward(self, inpt):
        self.scale = np.max(inpt)
        return inpt / self.scale

    def backward(self, d_next):
        return d_next / self.scale, 0, 0