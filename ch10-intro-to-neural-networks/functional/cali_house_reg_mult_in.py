import tensorflow as tf
from tensorflow import keras
import sys
import pathlib
import os

sys.path.append(os.path.join(pathlib.Path().absolute(), "sequential"))
from cali_house_reg import get_housing

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()
    tf.random.set_seed(42)

    # To allow different subsets to go down different paths
    # use multiple outputs. E.g. sending features 0-4 down
    # wide path, and 2-7 down the deep path. 
    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name="deep_input")

    # Hidden layers deal with input through B
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

    # Input through A and transformations on B are concatenated
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)

    model = keras.Model(inputs=[input_A, input_B], outputs=[output])
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    # Due to multi-inputs, datasets must be divided up into
    # subsets, and passed when training, evaluating, testing,
    # and predicting
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_val_A, X_val_B = X_val[:, :5], X_val[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

    history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                        validation_data=((X_val_A, X_val_B), y_val))
    
    print(model.evaluate((X_test_A, X_test_B), y_test))

    y_pred = model.predict((X_new_A, X_new_B))
    print(y_pred)