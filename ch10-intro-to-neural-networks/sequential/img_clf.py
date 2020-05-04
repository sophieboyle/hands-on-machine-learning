import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def plot_history(hist):
    pd.DataFrame(hist).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_fashion()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                    "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                    "Ankle boot"]
    
    # Building a classification MLP with 2 hidden layers
    model = keras.models.Sequential()
    # Flatten layer converts each image into a 1D array (Preproc.)
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # Layer with 300 neurons using ReLU activation function
    model.add(keras.layers.Dense(300, activation="relu"))
    # Layer with 100 neurons using ReLU activation function
    model.add(keras.layers.Dense(100, activation="relu"))
    # Output layer, 10 neurons (1 per class), using softmax
    # due to the exclusiveness of the classes 
    model.add(keras.layers.Dense(10, activation="softmax"))

    # The above is equivalent to the following:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])
    
    print(model.summary())

    # Investigating the details of the 1st hidden layer
    hidden1 = model.layers[1]
    print(hidden1.name)
    weights, biases = hidden1.get_weights()
    print(weights)
    print(weights.shape)
    print(biases)
    print(biases.shape)

    # Compile to specify loss function and optimiser
    # NOTE: Spare categorical crossentropy is used because the
    # labels are sparse and exclusive. Stochastic Gradient Descent
    # is specified to use backpropagation. (Uses default learning rate)
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="sgd", metrics=["accuracy"])

    # Training the model
    history = model.fit(X_train, y_train, epochs=30, 
                        validation_data=(X_val, y_val))

    # History now stores data regarding training parameters,
    # the list of epochs, and the loss and other metrics measured
    # at the end of each epoch.
    plot_history(history.history)

    # Evaluating on test set
    print(model.evaluate(X_test, y_test))

    # Predictions can be made on new instances
    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))

    # To predict the class with the highest probability
    y_pred = model.predict_classes(X_new)
    print(np.array(class_names)[y_pred])