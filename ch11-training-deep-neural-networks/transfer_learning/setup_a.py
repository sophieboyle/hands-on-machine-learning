import tensorflow as tf
from tensorflow import keras


def gen_model_a(X_train_A, y_train_A, X_val_A, y_val_A, X_test_A, y_test_A):
    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    # Adding 5 hidden layers with selu activation function
    for n_hidden in (300, 100, 50, 50, 50):
        model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    # 8 ouptut neurons as model A only identifies 8 classes
    model_A.add(keras.layers.Dense(8, activation="softmax"))

    model_A.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])
    
    history = model_A.fit(X_train_A, y_train_A, epochs=20,
                            validation_data=(X_val_A, y_val_A))
    
    # Save the model
    model_A.save("model_A.h5")