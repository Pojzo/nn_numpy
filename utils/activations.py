import numpy as np

class Activations:
    @staticmethod
    def sigmoid(inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))
    
    @staticmethod
    def relu(inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)
    
    @staticmethod
    def sigmoid_derivative(inputs: np.ndarray) -> np.ndarray:
        return Activations.sigmoid(inputs) * (1 - Activations.sigmoid(inputs))

    @staticmethod
    def relu_derivative(inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1, 0)

    @staticmethod
    def forward(function_name: str, inputs: np.ndarray) -> np.ndarray:
        if function_name == 'sigmoid':
            return Activations.sigmoid(inputs)
        elif function_name == 'relu':
            return Activations.relu(inputs)
        else:
            return inputs
    
    @staticmethod
    def backward(function_name: str, inputs: np.ndarray) -> np.ndarray:
        if function_name == 'sigmoid':
            return Activations.sigmoid_derivative(inputs)
        elif function_name == 'relu':
            return Activations.relu_derivative(inputs)
        else:
            return inputs