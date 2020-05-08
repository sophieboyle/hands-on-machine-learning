import tensorflow as tf
from tensorflow import keras


class ReconstructingRegressor(keras.Model):
    """Reconstructing regressor which attempts to minimise
    reconstruction alongside an overall loss function.
    """
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        # DNN with 5 dense and 1 output layer
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                kernel_initializer="lecun_normal")
                        for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
    
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        # Extra layer to reconstruct inputs
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
    
    def call(self, inputs):
        Z = inputs
        # Inputs go through all hidden layers
        for layer in self.hidden:
            Z = layer(Z)

        # Result goes through reconstruction layer
        reconstruction = self.reconstruct(Z)

        # Reconstruction loss calculated
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))

        # Scales down and adds to list of losses, ensuring that reconstruction
        # loss has a lesser weight than the main loss.
        self.add_loss(0.05 * recon_loss)

        return self.out(Z)
        