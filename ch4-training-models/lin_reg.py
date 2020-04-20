from sklearn.linear_model import LinearRegression
import numpy as np


"""
    @brief Perform a prediction on new data
    using scikit-learn linear regression.
    @param X true x-axis data
    @param y true y-axis data
    @param X_new data to predict based upon
    @return Array of prediction based on X_new
"""
def do_lin_reg_pred(X, y, X_new):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(f"Intercept: {lin_reg.intercept_}")
    print(f"Coef: {lin_reg.coef_}")
    prediction = lin_reg.predict(X_new)
    print(f"Prediction: {prediction}")
    return prediction


"""
    @brief Use least squares (i.e. via the pseudoinverse of X)
    to calculate the best model parameter. (This is what scikit-learn's
    linear regression model is based off of.)
    @param X numpy array of data
    @param y numpy array of labels
    @return theta, the best model parameter
"""
def lin_reg_manual_best_theta(X, y):
    # Use least squares to find theta, which computes theta via:
    # theta = X^+.y (X^+ is the pseudoinverse of X)
    theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=1e-6)
    print(theta)
    # Directly exemplify the above noted equation
    theta = np.linalg.pinv(X).dot(y)
    print(theta)
    # NOTE: The pseudoinverse is calculated using Singular Value
    # Decomposition. This is more efficient than the normal equation.
    return theta