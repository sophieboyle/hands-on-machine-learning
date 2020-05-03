import tensorflow as tf
from tensorflow import keras

def get_fashion():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    # Splitting training and validation and scaling pixel intensities
    # to values between 0 and 1 (instead of 255 different values)
    X_val = X_train_full[:5000] / 255.0 
    X_train = X_train_full[5000:] / 255.0
    y_val = y_train_full[:5000]
    y_train = y_train_full[5000:]
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_fashion()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                    "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                    "Ankle boot"]
    
    # Building a classification MLP with 2 hidden layers
    model = keras.models.Sequential()
    # Flatten layer converts each image into a 1D array (Preproc.)
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # Layer with 300 neurons using ReLU activation function
    model.add(keras.layers.Dense(300, activation="relu"))
    # Layer with 100 neurons using ReLU activation function
    model.add(keras.layers.Dense(100, activation="relu"))
    # Output layer, 10 neurons (1 per class), using softmax
    # due to the exclusiveness of the classes 
    model.add(keras.layers.Dense(10, activation="softmax"))

    # The above is equivalent to the following:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])
    
    print(model.summary())

    # Investigating the details of the 1st hidden layer
    hidden1 = model.layers[1]
    print(hidden1.name)
    weights, biases = hidden1.get_weights()
    print(weights)
    print(weights.shape)
    print(biases)
    print(biases.shape)

    # Compile to specify loss function and optimiser
    # NOTE: Spare categorical crossentropy is used because the
    # labels are sparse and exclusive. Stochastic Gradient Descent
    # is specified to use backpropagation.
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="sgd", metrics=["accuracy"])