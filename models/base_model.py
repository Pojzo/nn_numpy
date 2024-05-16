from abc import ABC, abstractmethod
import sys; sys.path.append("..")

from layers import Layer, Dense

class Model(ABC):
    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn
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
            output = layer.forward(output)

        return output

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