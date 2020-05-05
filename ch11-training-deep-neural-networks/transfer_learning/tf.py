import tensorflow as tf
from tensorflow import keras
from setup_data import *
from setup_a import *
import numpy as np

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_val, X_test, y_train, y_val, y_test = get_fashion()
    (X_train_A, y_train_A), (X_train_B, y_train_B) = split_data(
                                                        X_train, y_train)
    (X_val_A, y_val_A), (X_val_B, y_val_B) = split_data(X_val, y_val)
    (X_test_A, y_test_A), (X_test_B, y_test_B) = split_data(X_test, y_test)

    # Generate the base model
    gen_model_a(X_train_A, y_train_A, X_val_A, y_val_A, X_test_A, y_test_A)

    # Loading base model
    model_A = keras.models.load_model("model_A.h5")

    # Clone model (so as to not affect model_A's layers)
    model_A_clone = keras.models.clone_model(model_A)
    model_A_clone.set_weights(model_A.get_weights())

    # Reusing all layers except output layer
    model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])

    # Adding new output layer with one neuron (to identiy whether
    # sandal or shirt)
    model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

    # To avoid wrecking the lower level weights whilst the new layer
    # spends early epochs adjusting weights, freeze lower layers
    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = False

    # NOTE: Default learning rate for SGD is 1e-2
    model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
                        metrics=["accuracy"]) 
    
    # Now train for a few epochs with the frozen hidden layers
    history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                                validation_data=(X_val_B, y_val_B))
    
    # Now unfreeze the frozen layers
    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = True
    
    # Re-compile before retraining with a lower learning rate
    # to avoid wrecking the lower layers' weights
    optimizer = keras.optimizers.SGD(lr=1e-4) 
    model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                        metrics=["accuracy"])
    
    # Train for a few more epochs
    history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                                    validation_data=(X_val_B, y_val_B))

    # Transfer learning is actually pretty garbage on shallow nets
    # such as these; they are more effective on deep nets. 
    print(model_B_on_A.evaluate(X_test_B, y_test_B))