import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
    @brief Generates linear data to test
    the normal function.
"""
def gen_lin_data():
    # Random floats used for noise
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    # Plot the linear data
    plt.plot(X, y, 'ro')
    plt.show()
    return X, y


"""
    @brief Use the closed form (normal equation)
    to compute the best model parameter.
    @param X numpy data along the x axis
    @param y numpy data along the y axis
    @return Best model parameter.
"""
def compute_best_model_param(X, y):
    # Adding x0 = 1 to all instances
    X_b = np.c_[np.ones((100, 1)), X]
    # Compute the best model param using the normal equation
    best_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return best_theta


"""
    @brief Given a best model parameter and a feature vector
    perform a prediction.
    @param Theta: a numpy array of bias term and feature weights.
    @param X_b: a numpy array representing the feature vector.
    @return Numeric prediction.
"""
def predict_using_theta(theta, X_b):
    # Predict using prediction equation y^ = x . theta
    y_predict = X_b.dot(theta)
    return y_predict


"""
    @brief Using a model parameter, predict y
    on example data.
    @param Best model parameter
    @return Prediction
    @return X_b New data to fit
"""
def ex_predict_using_theta(theta):
    # New matrice: 2 rows, 1 col
    X = np.array([[0], [2]])
    # Adding x0 = 1 to all instances
    X_b = np.c_[np.ones((2, 1)), X]
    return predict_using_theta(theta, X_b), X


"""
    @brief Plot the true data against predictions
    @param X numpy array of true x-axis data
    @param y numpy array of true y-axis data
    @param X_fit numpy array of data used for predictions
    @param y_pred numpy array of resultant predictions
"""
def plot_predictions(X, y, X_fit, y_pred):
    plt.plot(X_fit, y_pred, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()