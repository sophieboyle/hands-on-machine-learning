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

    # Layout for both wide and deep paths
    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])

    # Multiple outputs: useful for many reasons such as
    # - Doing both regression and classification
    # - Multiclass classification
    # - Regularisation (an auxiliary output ensures that
    # underlying areas of the network learn something without
    # relying too heavily on the other sections.) 
    output = keras.layers.Dense(1, name="main_output")(concat)
    aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)

    model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
    
    # When compiling, a loss function must be set for each output.
    # The weight for the main output's loss function is greater,
    # as it is the output of greatest value. 
    model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1],
                    optimizer="sgd")

    # Dividing up data for multi-input
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_val_A, X_val_B = X_val[:, :5], X_val[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

    # When training, both multiple inputs and multiple output labels
    # must be provided 
    history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                        validation_data=([X_val_A, X_val_B], [y_val, y_val]))
    
    # Inidividual losses are returned in evaluation
    total_loss, main_loss, aux_loss = model.evaluate(
                                    [X_test_A, X_test_B], [y_test, y_test])
    print(total_loss, main_loss, aux_loss)

    # Multiple predictions are also returned
    y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
    print(y_pred_main, y_pred_aux)