import os
import gzip
import numpy as np

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)
    
def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_labels)

def load_fashion_mnist(data_path='data', flatten=True, normalize=True):
    """
    Loads Fashion-MNIST dataset from binary files.
    
    Parameters:
    - data_path: path to the folder containing .gz files
    - flatten: if True, images are flattened into 784-dim vectors
    - normalize: if True, pixel values are scaled to [0, 1]
    
    Returns:
    - x_train, y_train, x_test, y_test as numpy arrays
    """
    # Paths
    train_images = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
    train_labels = os.path.join(data_path, 'train-labels-idx1-ubyte.gz')
    test_images = os.path.join(data_path, 't10k-images-idx3-ubyte.gz')
    test_labels = os.path.join(data_path, 't10k-labels-idx1-ubyte.gz')
    
    # Load
    x_train = load_images(train_images)
    y_train = load_labels(train_labels)
    x_test = load_images(test_images)
    y_test = load_labels(test_labels)
    
    # Preprocess
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    if flatten:
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)
    
    return x_train, y_train, x_test, y_test