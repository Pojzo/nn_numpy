from abc import ABC, abstractmethod
import sys; sys.path.append("..")
import numpy as np

from layers import Layer, Dense

class Model(ABC):
    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn
        self.dtype = np.float64
        self.layers = []

    def add(self, layer):
        if type(layer) == Layer: 
            self.layers.append(layer)
        elif type(layer) == list:
            self.layers.extend(layer)
        else:
            raise Exception("Invalid layer")

    def forward(self, inpt):
        output = inpt
        for layer in self.layers:
            if isinstance(layer, Dense):
                output = layer.forward(output)

        return output
    
    def set_dtype(self, dtype):
        self.dtype = dtype
        for layer in self.layers:
            layer.set_dtype(dtype)
    
    def get_num_params(self):
        return sum([layer.get_num_params() for layer in self.layers])

    @abstractmethod
    def backward(self, inpt):
        pass

    @abstractmethod
    def train(self, inpt, target):
        pass

    @abstractmethod
    def predict(self, inpt):
        return self.forward(inpt)

    def __call__(self, inpt):
        return self.forward(inpt)
    
    def __repr__(self) -> str:
        return str(self.layers)