import tensorflow as tf
from tensorflow import keras


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