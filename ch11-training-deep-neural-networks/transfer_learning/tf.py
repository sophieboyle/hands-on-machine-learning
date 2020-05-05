import tensorflow as tf
from tensorflow import keras
from transfer_learning.setup_data import *
import numpy as np

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_val, X_test, y_train, y_val, y_test = get_fashion()
    (X_train_A, y_train_A), (X_train_B, y_train_B) = split_data(
                                                        X_train, y_train)
    (X_val_A, y_val_A), (X_val_B, y_val_B) = split_data(X_val, y_val)
    (X_test_A, y_test_A), (X_test_B, y_test_B) = split_data(X_test, y_test)
    