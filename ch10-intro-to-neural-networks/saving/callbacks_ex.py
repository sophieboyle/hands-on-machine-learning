import tensorflow as tf
from tensorflow import keras
import sys
import pathlib
import os

sys.path.append(os.path.join(pathlib.Path().absolute(), "sequential"))
from cali_house_reg import get_housing


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    """Example of custom callback, by printing the ratio of validation
    loss to training loss at the end of each epoch.

    Arguments:
        keras {Callback} -- Callback class.
    """
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"]/logs["loss"]))


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

    # Saving checkpoints of the model at the end of each epoch
    # Setting save_best_only means that only models with a new best
    # performance on the val set are saved.
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                    save_best_only=True)

    # Implements early stopping according to a patient param,
    # determining how many no-progress epochs to stop at. 
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                restore_best_weights=True)

    # Both checkpoint saving and early stopping can be combined
    history = model.fit(X_train, y_train, epochs=10, 
                        callbacks=[checkpoint_cb, early_stopping_cb])

    # Loading the best model only if early stopping were not being used
    # as early stopping automatically restores the best weights at the
    # end of training.
    # model = keras.models.load_model("my_keras_model.h5")