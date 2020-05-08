import tensorflow as tf
from tensorflow import keras


class CustDense(keras.layers.Layers):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, batch_input_shape):
        # Builds the layer by setting variables
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal"
        )
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros"
        )
        super.build(batch_input_shape)
    
    def call(self, X):
        # Compute matrice mult of X and kernel, add the bias, and
        # return the result of the activation function on the computation
        return self.activation(X @ self.kernel + self.bias)
    
    def compute_output_shape(self, batch_input_shape):
        # Gives shape of outputs: the input shape with the last dimension
        # being the number of neurons.
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        # NOTE: The activation function's entire config is 
        # saved using serialize
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


class CustMultiLayers(keras.layers.Layer):
    def call(self, X):
        # Multi-input: X is a tuple of inputs
        X1, X2 = X
        # Multi-outputs: return list from call()
        return [X1 + X2, X1 * X2, X1 / X2]
    
    def compute_output_shape(self, batch_input_shape):
        # Multi-inpt: batch_input_shape is a tuple containing all
        # input batch shapes. 
        b1, b2 = batch_input_shape
        # Multi-ouptut: list of batch output shapes
        return [b1, b1, b1]