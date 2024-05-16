from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, activation='sigmoid'):
        self.activation = activation
        
    @abstractmethod
    def forward(self, inputs):
        pass
    @abstractmethod
    def backward(self, outputs):
        pass