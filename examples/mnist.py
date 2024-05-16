import sys, os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClassifierModel
from layers import Dense, Flatten, Normalize

from examples.utils import download_mnist

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


(train_images, train_labels), (test_images, test_labels) = download_mnist('data')

import pickle
import os

model_path = 'model.pkl'

def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print("Loaded model from disk.")
            test_acc = model.test(test_images, np.eye(10)[test_labels])
            train_acc = model.test(train_images, np.eye(10)[train_labels])
            print(f"Test accuracy: {test_acc}")
            print(f"Train accuracy: {train_acc}")
            return model
    else:
        return None
def main():
    parse = argparse.ArgumentParser(description='Train a new model or load an existing one.')
    parse.add_argument('--train', action='store_true', help='Train a new model.', default=False)
    parse.add_argument('--resume', action='store_true', help='Resume training.', default=False)
    args = parse.parse_args()

    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print("Loaded model from disk.")
        if args.train and not args.resume:
            print("A model already exists, but a new one will be trained.")
            model = train_new_model()
        elif not args.train and args.resume:
            print("Resuming training of existing model.")
            model = resume_training(model)
        elif args.train and args.resume:
            print("A model already exists, but a new one will be trained and then training will be resumed.")
            model = train_new_model()
            model = resume_training(model)
        else:
            test_acc = model.test(test_images, np.eye(10)[test_labels])
            train_acc = model.test(train_images, np.eye(10)[train_labels])
            print(f"Test accuracy: {test_acc}")
            print(f"Train accuracy: {train_acc}")
    else:
        if args.train or args.resume:
            print("No existing model found. A new one will be trained.")
            model = train_new_model()
            if args.resume:
                print("Resuming training of new model.")
                model = resume_training(model)
        else:
            print("No existing model found and no training argument provided. Please provide either --train or --resume argument.")
            return

def train_new_model():
    model = ClassifierModel()

    activation = 'sigmoid'

    dense1 = Dense(28 * 28, 256, activation='relu')
    dense2 = Dense(256, 128, activation='relu')
    dense3 = Dense(128, 64, activation='relu')
    dense4 = Dense(64, 32, activation='sigmoid')
    dense5 = Dense(32, 10, activation='sigmoid')
    flatten = Flatten()
    normalize = Normalize()

    model.add([flatten, normalize, dense1, dense2, dense3, dense4, dense5])
    model.start_loss_history()
    model.start_acc_history()

    n_samples = 60000

    data_x = train_images[:n_samples]
    data_y = train_labels[:n_samples]

    print(data_x.shape, data_y.shape)

    model.train(data_x, data_y, batch_size = 256, epochs=100, print_freq=10)
    plt.plot(model.losses)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Saved model to disk.")
    return model

def resume_training(model):
    n_samples = 60000
    data_x = train_images[:n_samples]
    data_y = train_labels[:n_samples]
    model.train(data_x, data_y, batch_size = 256, epochs=100, print_freq=10)
    plt.plot(model.losses)
    return model

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()