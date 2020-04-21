from sklearn.linear_model import Ridge, SGDRegressor
import numpy as np


def gen_reg_ex_data():
    np.random.seed(42)
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)
    return X, y


"""
    @brief Perform ridge regression using cholesky's
    matrix factorisation with the closed-form equation.
    @param X data array of feature values.
    @param y data array of labels.
    @return Fitted ridge regressor.
"""
def do_ridge_reg(X, y):
    # Perform ridge regression using the matrix factorisation
    # technique by Choleski
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    return ridge_reg


"""
    @brief Perform ridge regression using Stochastic
    Gradient Descent.
    @param X data array of feature values.
    @param y data array of labels.
    @return Fitted SGD Regressor.
"""
def do_ridge_reg_using_sgd(X, y):
    # Penalty specifies the type of regularisation term
    # l2: add half the square of the l2 norm of the weight
    # vector to the cost function.
    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    return sgd_reg