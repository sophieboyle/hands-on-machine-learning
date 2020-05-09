import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def random_batch(X, y, batch_size=32):
    """Samples a random batch from the dataset.

    Arguments:
        X {nparray} -- Dataset
        y {nparray} -- Labels

    Keyword Arguments:
        batch_size {int} -- Number of instances to sample (default: {32})

    Returns:
        nparray -- Dataset sample
        nparray -- Labels sample
    """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_status_bar(iteration, total, loss, metrics=None):
    # Outputs status bar
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result)
                    for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} -".format(iteration, total) + metrics, end=end)


def get_housing():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
                            X_train_full, y_train_full, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # Get data
    X_train, X_val, X_test, y_train, y_val, y_test = get_housing()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Building a simple model
    l2_reg = keras.regularizers.l2(0.05)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                            kernel_regularizer=l2_reg),
        keras.layers.Dense(1, kernel_regularizer=l2_reg)
    ])

    # Setting up hyperparams, optimizer, loss funct, and metrics
    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.MeanAbsoluteError()]

    # Custom training loop
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")

        # Goes over each batch in the epoch
        for step in range(1, n_steps):
            # Sample batch from training set
            X_batch, y_batch = random_batch(X_train_scaled, y_train)

            with tf.GradientTape() as tape:
                # Make predictions and compute the loss over the batch
                y_pred = model(X_batch, training=True)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            
            # Compute gradient of loss in regards to all trainable vars
            # and apply them to optimizer 
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update mean loss and metrics
            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
            
            # Reset states
            for metric in [mean_loss] + metrics:
                metric.reset_states()
