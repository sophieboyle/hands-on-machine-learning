import tensorflow as tf
from tensorflow import keras
import numpy as np


def random_batch(X, y, batch_size=32):
    """Samples a random batch from the dataset.

    Arguments:
        X {nparray} -- Dataset
        y {nparray} -- Labels

    Keyword Arguments:
        batch_size {int} -- Number of instances to sample (default: {32})

    Returns:
        nparray -- Dataset sample
        nparray -- Labels sample
    """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


if __name__ == "__main__":
    # Building a simple model
    l2_reg = keras.regularizers.l2(0.05)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                            kernel_regularizer=l2_reg),
        keras.layers.Dense(1, kernel_regularizer=l2_reg)
    ])
