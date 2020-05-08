import tensorflow as tf
from tensorflow import keras


class ResidualBlock(keras.layers.Layer):
    """ResidualBlock layer which contains multiple other
    dense layers, aggregating the outputs of each dense
    layer, and then adding the final output to the input.
    """
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        # List of dense layers to generate
        # Hidden automatically detects that trackable objects are contained
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                        kernel_initializer="he_normal")
                        for _ in range(n_layers)]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            # Accumulates outputs of each layer
            Z = layer(Z)
        # Adds inputs to the final outputs of the dense layers
        return inputs + Z


class ResidualRegressor(keras.Model):
    """Residual Regressor custom model, which contains
    a dense layer, 3 repetitions of the first residual
    block, another residual block, and a final dense layer.
    """
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu",
                                    kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)
    
    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)