import tensorflow as tf
from tensorflow import keras
import sys
import pathlib
import os

sys.path.append(os.path.join(pathlib.Path().absolute(), "sequential"))
from cali_house_reg import get_housing

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()
    tf.random.set_seed(42)

    # Input object to specify models input shape (and type)
    input_ = keras.layers.Input(shape=X_train.shape[1:])

    # Create two hidden layers, calling them each as a function
    # and passing them the ouptut of the previous layer it connects to 
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

    # Concatenate layer connects to two layers
    concat = keras.layers.Concatenate()([input_, hidden2])

    # Output layer with one neuron to output a value
    output = keras.layers.Dense(1)(concat)
    
    # Creates the model with specified input and output
    model = keras.Model(inputs=[input_], outputs=[output])

    # Training, compilation, and predicting works the same as before
    model.compile(loss="mean_squared_error", optimizer="sgd")
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_val, y_val))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print(y_pred)