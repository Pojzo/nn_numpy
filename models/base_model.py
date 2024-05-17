from abc import ABC, abstractmethod
import sys; sys.path.append("..")
import numpy as np

from layers import Layer, Dense

class Model(ABC):
    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn
        self.dtype = np.float64
        self.batch_size = 256
        self.lr = 0.01
        self.epochs = 100
        self.name = "model"
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
    
    def save_hyperparameters(self, **kwargs):
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        
        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        
        if 'epochs' in kwargs:
            self.epochs = kwargs['epochs']
        
        if 'name' in kwargs:
            self.name = kwargs['name']
        
    def get_lr(self):
        return self.lr
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_epochs(self):
        return self.epochs
    
    def get_name(self):
        return self.name

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