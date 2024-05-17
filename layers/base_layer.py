from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, activation='sigmoid'):
        self.activation = activation
        
    @abstractmethod
    def forward(self, inputs):
        pass
    @abstractmethod
    def backward(self, outputs, lr=0.01):
        pass

    def set_dtype(self, dtype):
        if hasattr(self, 'weights'):
            self.weights = self.weights.astype(dtype)

    def get_num_params(self):
        return 0