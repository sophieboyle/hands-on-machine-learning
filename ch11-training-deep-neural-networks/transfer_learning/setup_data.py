import tensorflow as tf
from tensorflow import keras
import numpy as np


def split_data(X, y):
    """Segments dataset into different subsets consisting
    of all items except sandals/shirts, and only sandals/shirts.
    Expects the MNIST fashion dataset.

    Arguments:
        X {nparray} -- Data for MNIST fashion
        y {nparray} -- Labels for MNIST fashion
    Returns:
        Tuple -- Dataset and labels for all items excluding
                 sandals and shirts
        Tuple -- Dataset and labels for sandals and shirts
    """
    # Sandals or shirt identifier
    y_5_or_6 = (y == 5) | (y == 6)

    # Gets all items that are not sandals/shirts
    y_A = y[~y_5_or_6]

    # Class indices in first subset must be shifted down
    # to fill the gap of having no sandals/shirts. 
    y_A[y_A > 6] -= 2

    # Turns being a sandal or a shirt into binary classification
    y_B = (y[y_5_or_6] == 6).astype(np.float32)

    return ((X[~y_5_or_6], y_A), (X[y_5_or_6], y_B))



def get_fashion():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    # Splitting training and validation and scaling pixel intensities
    # to values between 0 and 1 (instead of 255 different values)
    X_val = X_train_full[:5000] / 255.0 
    X_train = X_train_full[5000:] / 255.0
    y_val = y_train_full[:5000]
    y_train = y_train_full[5000:]
    return X_train, X_val, X_test, y_train, y_val, y_test