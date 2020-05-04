import tensorflow as tf
from tensorflow import keras
import sys
import os
import pathlib
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


sys.path.append(os.path.join(pathlib.Path().absolute(), "sequential"))
from cali_house_reg import get_housing


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    """Creates a MLP model for univariate regression, given the parameters.

    Keyword Arguments:
        n_hidden {int} -- Number of hidden layers (default: {1})
        n_neurons {int} -- Number of neurons per layer (default: {30})
        learning_rate {float} -- Learning rate of GradDesc alg (default: {3e-3})
        input_shape {list} -- Shape of the input data (default: {[8]})

    Returns:
        Sequential MLP Model -- A sequential MLP compiled with the params.
    """
    model = keras.models.Sequential()

    # Adds input layer according to given input shape
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # Initialises each hidden layer with the given number of neurons
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    
    # Adds output layer with one neuron
    model.add(keras.layers.Dense(1))

    # Initialisation stochastic gradDesc as optimiser with the given
    # learning rate 
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()
    X_new = X_test[:3]

    # Thin wrapper around the keras model built using build model with
    # default parameters.
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    # Therefore, methods can be used as if it were a scikit-learn reg.
    keras_reg.fit(X_train, y_train, epochs=100, 
                    validation_data=(X_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # NOTE: The score will be the opposite of MSE, as scikit-learn's
    # score method works by providing higher values for better models.
    mse_test = keras_reg.score(X_test, y_test)
    print(f"SCORE: {mse_test}")

    y_pred = keras_reg.predict(X_new)
    print(y_pred)

    # Doing a randomised search on hyperparams (as there are many)
    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2)
    }

    # Uses k-fold validation (i.e. not touching validation sets)
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                        n_iter=10, cv=3)

    # Uses the validation sets for early stopping only
    rnd_search_cv.fit(X_train, y_train, epochs=100, 
                        validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    
    print(f" Best params: {rnd_search_cv.best_params_}")
    print(f"Best params: {rnd_search_cv.best_score_}")
    
    # Selecting the best model
    model = rnd_search_cv.best_estimator_.model