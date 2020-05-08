import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def print_status_bar(iteration, total, loss, metrics=None):
    # Outputs status bar
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result)
                    for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} -".format(iteration, total) + metrics, end=end)


def get_housing():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
                            X_train_full, y_train_full, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # Get data
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()

    # Building a simple model
    l2_reg = keras.regularizers.l2(0.05)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                            kernel_regularizer=l2_reg),
        keras.layers.Dense(1, kernel_regularizer=l2_reg)
    ])

    # Setting up hyperparams, optimizer, loss funct, and metrics
    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.MeanAbsoluteError()]