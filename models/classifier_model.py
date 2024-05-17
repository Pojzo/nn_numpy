from models import Model

import numpy as np
import time

from losses import MSE
from layers import Flatten, Normalize

"""
ClassifierModel class that extends the Model class.
"""
class ClassifierModel(Model):
    def __init__(self, loss_fn=MSE()):
        super().__init__(loss_fn=loss_fn)
        self.accs = None
        self.losses = None

    """
    Overrides the backward method of the Model class
    Computes the loss and accuracy, and backpropagates the gradients
    """
    def backward(self, one_hot_target, target_argmax):
        pred = self.layers[-1].output
        loss = self.loss_fn(pred, one_hot_target)

        d_loss = self.loss_fn.backward(pred, one_hot_target)

        dl_dx = d_loss
        for layer in self.layers[::-1]:
            dl_dx, _, _ = layer.backward(dl_dx)

        if self.losses is not None:
            self.losses.append(np.mean(loss))

        pred_argmax = np.argmax(pred, axis=1)
        # target_argmax = np.argmax(target, axis=1)
        acc = np.sum(pred_argmax == target_argmax) / len(pred_argmax)
        
        if self.accs is not None:
            self.accs.append(acc)

        return loss, acc
    
    """
    Preprocess the data by passing it through the first layers of the model,
    which are assumed to be Flatten and Normalize layers to avoid having to 
    compute the outputs of these layers on every forward pass.
    """
    def preprocess_data(self, data_x):
        for layer in self.layers:
            if isinstance(layer, Flatten) or isinstance(layer, Normalize):
                data_x = layer.forward(data_x)
            else:
                return data_x
        
        return data_x

    """
    Train the model on the given data.
    """
    def train(self, data_x, data_y, epochs=100, batch_size=32, print_freq=10):
        start_time = time.time()

        one_hot_target = np.eye(10)[data_y]
        num_samples = data_x.shape[0]

        data_x = self.preprocess_data(data_x)
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = []
            num_batches = num_samples // batch_size
            for batch_start in range(0, num_samples, batch_size):
                
                batch_end = min(batch_start + batch_size, num_samples)
                batch_x = data_x[batch_start:batch_end]
                batch_y = one_hot_target[batch_start:batch_end]

                output= self.forward(batch_x)

                loss, acc = self.backward(batch_y, data_y[batch_start:batch_end])
                epoch_loss += loss.sum()
                epoch_acc += [acc]            
            # Average loss for the epoch
            epoch_loss /= num_samples
            epoch_acc = np.mean(epoch_acc)

            # Print progress
            if (epoch + 1) % print_freq == 0:
                avg_epoch_time = (time.time() - start_time) / (epoch + 1)
                print(f"Epoch {epoch + 1}/{epochs}: Loss - {epoch_loss:.4f}. Acc - {epoch_acc:.4f}. Avg epoch time: {avg_epoch_time:.4f}s")

        total_time = time.time() - start_time
        print(f"Finished training for {epochs} epochs: Final Loss - {epoch_loss:.4f}. Acc - {epoch_acc:.4f}. Total time: {total_time:.4f}s")

    """
    Predict the output of the model for the given input.
    """
    def predict(self, inpt):
        return super().predict(inpt)

    """
    Test the model on the given data.
    """
    def test(self, data_x, data_y):
        data_x = self.preprocess_data(data_x)
        pred = self(data_x)
        pred_argmax = np.argmax(pred, axis=1)

        target_argmax = np.argmax(data_y, axis=1)

        return np.sum(target_argmax == pred_argmax) / len(pred_argmax)

    def get_stats(self, data_x, data_y):
        best_acc_train = np.max(self.accs)
        acc_test = self.test(data_x, data_y)

        return {'acc_train': best_acc_train, 'acc_test': acc_test}

    def get_wrong_predictions(self, data_x, data_y):
        pred = self(data_x)
        
    def start_loss_history(self):
        self.losses = []

    def start_acc_history(self):
        self.accs = []

    def get_loss_history(self):
        return self.losses

    def get_acc_history(self):
        return self.accs