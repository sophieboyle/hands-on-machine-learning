from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


def get_housing():
    # Simpler version of the data, with no missing values and
    # only numerical features. 
    housing = fetch_california_housing()

    # Divide up datasets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
                                        housing.data, housing.target,
                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full,
                                                    y_train_full,
                                                    random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()

    # One hidden layer since the dataset is noisy (avoids overfitting)
    # Output layer only has 1 neuron, as it must output one value
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", 
                            input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])

    # Specifying loss function and optimiser for backprop.
    model.compile(loss="mean_squared_error", optimizer="sgd")

    # Fitting the model
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_val, y_val))
    
    # Compute MSE on the test data
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)

    # Making new predictions
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print(y_pred)