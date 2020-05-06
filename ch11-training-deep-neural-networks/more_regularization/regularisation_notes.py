import tensorflow as tf
from tensorflow import keras
from functools import partial

if __name__ == "__main__":
    # Sets up layer with l2 norm regularization, which contrains the
    # connection weights. 
    layer = keras.layers.Dense(100, activation="elu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=keras.regularizers.l2(0.01))

    # Sets up layer with l1 norm regularization, which helps to produce
    # a sparse model.
    layer = keras.layers.Dense(100, activation="elu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=keras.regularizers.l1(0.01))

    # Creates a wrapper for callable with default arguments
    # This is useful when wanting to apply the same activation function
    # and regularisation to many layers without repeating code or loops 
    RegularisedDense = partial(keras.layers.Dense,
                                activation="elu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=keras.regularizers.l2(0.01))

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        RegularisedDense(300),
        RegularisedDense(100),
        RegularisedDense(10, activation="softmax", 
                        kernel_initializer="glorot_uniform")
    ])

    # Applying Dropout to a model, where on each step, each neuron
    # is subject to a dropout probability, which is the probability
    # of it being completely ignored for this step. (Also involves
    # multiplying the input by a keep probability.) This ensures
    # that neurons do not co-adapt and are all useful independently.
    # Since this trains a different NN at each step, it can be seen as
    # averaging an ensemble of several neural networks.
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])

    # NOTE: MCDropout can be used on top of trained dropout models
    # to boost the accuracy and provide the ability to investigate
    # the probabilities of predictions in further detail. 

    # Max norm regularisation involves constraining the weights of each
    # neuron such that they remain below a max-norm hyperparameter.
    # After each training step, w is rescalled to abide by this.
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
                        kernel_constraints=keras.constraints.max_norm(1.))