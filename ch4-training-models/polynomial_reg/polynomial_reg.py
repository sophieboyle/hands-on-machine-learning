import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


"""
    @brief Generate noisy data according to a
    quadratic equation (i.e. non-linear) and display
    the plot of this data.
    @return X array of data
    @return y array of labels
    @return m number of instances
"""
def gen_quadratic_data():
    m = 100
    # Use random to add noise
    np.random.seed(42)
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    # Display data
    plt.plot(X, y, 'ro')
    plt.show()
    return X, y, m


"""
    @brief Add the square of each feature in feature array
    X to the array, and return the transformed data.
    @param X the data to transform.
    @return The feature array with polynomial features.
    @return The fitted polynomialFeatures transformer.
"""
def add_poly_features(X):
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    return X_poly, poly_features


"""
    @brief Use the linear regression model to fit on polynomial
    data and return the predictions on X.
    @param X array of feature values
    @param y array of labels
    @param X_new_poly polynomial data to predict upon
    @return array of predictions using trained linear regression
    model.
"""
def fit_lin_reg_on_poly_data(X_poly, y, X_new_poly):
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    return lin_reg.predict(X_new_poly)


"""
    @brief Generate new X data with polynomial features
    of dimension (100, 2).
    @param poly_features Previously fitted transformer.
    @return New data.
    @return New data with poly features.
"""
def gen_new_X_data_for_pred(poly_features):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    return X_new, X_new_poly


"""
    @brief Plots predictions against original data.
    @param X feature value array
    @param y label array
    @param X_new feature value array to predict upon
    @param y_pred array of predictions made
"""
def plot_poly_pred(X, y, X_new, y_pred):
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_pred, "r-")
    plt.axis([-3, 3, 0, 10])
    plt.show()