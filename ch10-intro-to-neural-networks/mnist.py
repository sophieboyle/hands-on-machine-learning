import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class GrowLearningRateCallback(keras.callbacks.Callback):
    """Stores information
        regarding the learning rate and loss at each batch, whilst
        increasing the learning rate according to a given factor.

    Arguments:
        keras {Callback} -- Callback class
    """
    def __init__(self, factor):
        self.factor = factor
        self.loss = []
        self.lrs = []
    
    def on_batch_end(self, batch, logs):
        self.loss.append(logs["loss"])
        self.lrs.append(keras.backend.get_value(self.model.optimizer.lr))
        keras.backend.set_value(
            self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def get_mnist():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_val, X_train = X_train_full[:5000] / 225.0, X_train_full[5000:] / 225.0
    y_val, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 225.0
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_lr_loss(grow_lr_cb):
    """Plot the learning rates against losses stored
    in the GrowLearningRateCallback object.

    Arguments:
        grow_lr_cb {GrowLearningRateCallback} -- 
    """
    plt.plot(grow_lr_cb.lrs, grow_lr_cb.loss)
    plt.gca().set_xscale('log')
    plt.axis([min(grow_lr_cb.lrs), max(grow_lr_cb.lrs), 0, grow_lr_cb.loss[0]])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    # Preparation
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # Get MNIST data
    X_train, X_val, X_test, y_train, y_val, y_test = get_mnist()

    # Initialise model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3), 
                    metrics=["accuracy"])
    
    # Initialise callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint("mnist_mlp.h5",
                                                    save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                        restore_best_weights=True)
    
    grow_lr_cb = GrowLearningRateCallback(factor=1.005)

    # Train on just 1 epoch, growing the learning rate
    # at each batch. 
    history = model.fit(X_train, y_train, epochs=1, 
                        callbacks=[grow_lr_cb])

    # Plot learning growth vs loss
    plot_lr_loss(grow_lr_cb)

    # NOTE: Plot shows that loss seems to spike at 3e-1,
    # therefore, try a model with lr 2e-2

    keras.backend.clear_session()

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=2e-1), 
                    metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint_cb, early_stopping_cb])
    
    model = keras.models.load_model("mnist_mlp.h5")
    print(model.evaluate(X_test, y_test))