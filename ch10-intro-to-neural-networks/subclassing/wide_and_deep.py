import tensorflow as tf
from tensorflow import keras

class WideAndDeepModel(keras.Model):
    """Subclass API version of functional.cali_house_reg_mult_in_out
    model. NOTE: Keras cannot inspect subclass models as easily,
    therefore there may be issues caused by being unable to check
    shapes and types before runtime, alongside being unable to 
    determine the connections between layers using summary().

    Arguments:
        keras {Model} -- Keras Model class
    """
    def __init__(self, units=30, activation="relu", **kwargs):
        # Constructs the layers
        super.__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    
    def call(self, inputs):
        # Uses the layers
        # Separating usage from construction allows for
        # implementation of control flows. 
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output