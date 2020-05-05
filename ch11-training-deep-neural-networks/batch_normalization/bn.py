import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    # Implementing MLP with batch normalisation for all hidden
    # layers, and also the input layer. NOTE: BN may not show
    # much improvement for a shallow net, but tends to show
    # improvement for neural networks with more hidden layers.
    # Also uses ELU act. funct. for all hidden layers.
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, activation="elu", 
                            kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation="elu",
                            kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ]) 

    # It can be seen that each BN layers adds 4 params per input
    # for standardising, scaling, and offsetting. These params are:
    # the ouptut scale vector, output offset vector, final input
    # standard dev vector, and the final input mean vector.  
    # The latter "final" vectors are untrainable, as they are
    # moving averages which are unaffected by backpropagation.
    print(model.summary())

    # Implementing BN before each layer instead of after.
    # Also removing the bias term from the prev layer.
    # This is possible because each BN layer provides an offset
    # param per input
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("elu"),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("elu"),
        keras.layers.Dense(10, activation="softmax")
    ]) 