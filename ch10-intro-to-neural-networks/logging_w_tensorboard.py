import time
import os
import tensorflow as tf
from tensorflow import keras
from sequential.cali_house_reg import get_housing

root_logdir = os.path.join(os.curdir, "logs")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()
    tf.random.set_seed(42)

    # Building and compiling the simple sequential model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", 
                            input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="sgd")

    # Using TensorBoard callback
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_cb])