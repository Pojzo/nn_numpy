from .base_loss import Loss

import numpy as np

class MSE(Loss):
    def forward(self, pred, target):
        return np.square(np.subtract(pred, target)).mean(axis=1)

    def backward(self, pred, target):
        return 2 * (pred - target) / pred.shape[0]