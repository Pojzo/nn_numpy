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

get_final_path = lambda x: os.path.join("examples", "mnist", "saved_models", x)

doc_template = """
# Model Training Report

## Training

The model was trained using the following parameters:

- Epochs: {epochs}
- Batch size: {batch_size}
- Learning rate: {learning_rate}

## Training and Validation Statistics

The final training accuracy was {train_acc}%, and the final validation accuracy was {test_acc}%.

The loss decreased steadily over the training process, as shown in the following graph:

![Loss Graph](path_to_loss_graph.png)

The accuracy increased over the training process, as shown in the following graph:

![Accuracy Graph](path_to_accuracy_graph.png)
"""

model_path = 'model.pkl'

def load_model():
    saved_model_path = get_final_path(model_path)
    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
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
    parser = argparse.ArgumentParser(description='Train a new model or load an existing one.')
    parser.add_argument('--train', type=bool, default=False, help='Train a new model.')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to save the model.')
    args = parser.parse_args()

    saved_model_path = get_final_path(args.model_path)
    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
            model = pickle.load(f)
            print("Loaded model from disk.")
        if args.train and not args.resume:
            print("A model already exists, but a new one will be trained.")
            model = train_new_model(epochs=args.epochs)
        elif not args.train and args.resume:
            print("Resuming training of existing model.")
            model = resume_training(model, epochs=args.epochs)
        elif args.train and args.resume:
            print("A model already exists, but a new one will be trained and then training will be resumed.")
            model = train_new_model(epochs=args.epochs)
            model = resume_training(model, epochs=args.epochs)
        else:
            test_acc = model.test(test_images, np.eye(10)[test_labels])
            train_acc = model.test(train_images, np.eye(10)[train_labels])
            print(f"Test accuracy: {test_acc}")
            print(f"Train accuracy: {train_acc}")
            print(f"Number of parameters: {model.get_num_params()}")
    else:
        if args.train or args.resume:
            print("No existing model found. A new one will be trained.")
            model = train_new_model(epochs=args.epochs)
            if args.resume:
                print("Resuming training of new model.")
                model = resume_training(model, epochs=args.epochs)
        else:
            print("No existing model found and no training argument provided. Please provide either --train or --resume argument.")
            return

def train_new_model(epochs=100, batch_size=256, print_freq=10):
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

    model.set_dtype(np.float32)

    n_samples = 60000

    data_x = train_images[:n_samples].astype(np.float32)
    data_y = train_labels[:n_samples]

    model.train(data_x, data_y, batch_size=batch_size, epochs=epochs, print_freq=print_freq)

    gather_stats_and_save(model)

    # plt.plot(model.losses)

    # save_model(model)

    # return model

def resume_training(model, n_samples=60000, epochs=100, batch_size=256, print_freq=10):
    n_samples = 60000

    data_x = train_images[:n_samples]
    data_y = train_labels[:n_samples]

    model.train(data_x, data_y, batch_size=batch_size, epochs=epochs, print_freq=print_freq)

    gather_stats_and_save(model)

    # test_acc = model.test(test_images, np.eye(10)[test_labels])
    # train_acc = model.test(train_images, np.eye(10)[train_labels])

    # print(f"Test accuracy: {test_acc}")
    # print(f"Train accuracy: {train_acc}")
    # plt.plot(model.losses)
    # save_model(model)
    # return model

def save_model(model, doc_string=None, doc_path=None):
    final_path = get_final_path(model_path)
    with open(final_path, 'wb') as f:
        pickle.dump(model, f)
    print("Saved model to disk.", final_path)

    if doc_string and doc_path:
        with open(doc_path, 'a') as f:
            f.write(doc_string)
        print("Saved documentation to disk.", doc_path)

def gather_stats_and_save(model):
    global train_images, train_labels, test_images, test_labels
    train_data = train_images[:60000].astype(np.float32)
    train_labels = train_labels[:60000]

    test_data = test_images[:10000].astype(np.float32)
    test_labels = test_labels[:10000]

    train_acc = model.test(train_data, np.eye(10)[train_labels])
    test_acc = model.test(test_data, np.eye(10)[test_labels])

    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")

    lr = model.get_lr()
    batch_size = model.get_batch_size()
    epochs = model.get_epochs()
    name = model.get_name()

    final_string = doc_template.format(epochs=epochs, batch_size=batch_size, learning_rate=lr, train_acc=train_acc, test_acc=test_acc)

    final_path = get_final_path(name + ".pkl")
    documentation_path = os.path.join("examples", "mnist", "readme.md")

    save_model(model, doc_string=final_string, doc_path=documentation_path)

if __name__ == '__main__':
    main()