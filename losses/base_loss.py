from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, pred, target):
        pass

    @abstractmethod
    def backward(self, pred, target):
        pass

    def __call__(self, pred, target):
        return self.forward(pred, target)