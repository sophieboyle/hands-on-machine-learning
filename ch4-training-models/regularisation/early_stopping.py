from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


"""
    @brief Generate and divide example data
    @param m Integer number of instances.
    @return X_train feature value training array.
    @return X_val feature value validation array.
    @return y_train labels for training array.
    @return y_val labels for validation array.
"""
def gen_and_split_data(m):
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)
    return train_test_split(X[:50], y[:50].ravel(), test_size=0.5,
                            random_state=10)


"""
    @brief Add polynomial features to and scale feature
    value arrays for a training and validation set.
    @return Transformed training feature value array.
    @return Transformed validation feature value array.
"""
def prep_data(X_train, X_val):
    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False),
        ("std_scaler", StandardScaler()))
    ])
    return poly_scaler.fit_transform(X_train), poly_scaler.fit(X_val)


"""
    @brief Find the best SGDRegression model via early stopping.
    @param X_train feature value array of training instances.
    @param X_val feature value array of validation instances.
    @param y_train label array for training set.
    @param y_val label array for validation set.
    @return Best fitting SGDRegressor
"""
def early_stopping_sgdr(X_train, X_val, y_train, y_val):
    # Initialise SGDRegressor model
    sgd = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                        penalty=None, learning_rate="constant",
                        eta0=0.0005)
    # Initialise vars to save a model which has potentially
    # reached the minimum.
    minimum_val_err = float("inf")
    best_epoch = None
    best_model = None
    
    for epoch in range(1000):
        sgd.fit(X_train, y_train)
        y_val_pred = sgd.predict(X_val)
        val_error = mean_squared_error(y_val, y_val_pred)
        # If the validation error for the current model
        # is smaller than the minimum found thus far, update.
        if val_error < minimum_val_err:
            minimum_val_err = val_error
            best_epoch = epoch
            best_model = clone(sgd)
    
    return best_model