import numpy as np
import tensorflow as tf

import os

def download_mnist(data_dir):
    """Download and load the MNIST dataset if not already present in the specified directory."""
    # Define paths for train and test images and labels
    paths = [
        os.path.join(data_dir, 'mnist.npz')
    ]

    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if the dataset already exists
    if all(os.path.exists(path) for path in paths):
        print("MNIST dataset already exists in the specified directory.")
    else:
        print("Downloading MNIST dataset...")
        # Download the dataset using get_file
        dataset_path = tf.keras.utils.get_file(
            'mnist.npz',
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
            cache_dir=data_dir,
            cache_subdir=''
        )
        print(f"Dataset downloaded to: {dataset_path}")

    # Load the dataset
    with np.load(os.path.join(data_dir, 'mnist.npz')) as data:
        train_images = data['x_train']
        train_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']
    
    return (train_images, train_labels), (test_images, test_labels)