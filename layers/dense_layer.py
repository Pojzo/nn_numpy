import sys; sys.path.append("..")

from layers.base_layer import Layer

from utils.activations import Activations

import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, activation='sigmoid'):
        super().__init__(activation=activation)
        # use He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

        self.bias = np.random.rand(output_size)
        self.shape = self.weights.shape

    def forward(self, inputs):
        self.inputs = inputs
        self.z = inputs @ self.weights + self.bias

        if self.activation == 'sigmoid':
            self.output = Activations.sigmoid(self.z)
        elif self.activation == 'relu':
            self.output = Activations.relu(self.z)
        else:
            self.output = self.z

        return self.output

    def backward(self, d_next, lr=0.01):
        if self.activation == 'sigmoid':
            d = Activations.sigmoid_derivative(self.z) * d_next
        elif self.activation == 'relu':
            d = Activations.relu_derivative(self.z) * d_next
        else:
            d = d_next

        dl_dw = self.inputs.T @ d
        dl_db = np.sum(d, axis=0)
        dl_dx = d @ self.weights.T

        self.weights -= dl_dw * lr
        self.bias -= dl_db * lr

        return dl_dx, dl_dw, dl_db