from sklearn.linear_model import Lasso, SGDRegressor


"""
    @brief Perform Least Absolute Shrinkage and
    Selection Operator Regression. This adds l1 norm to 
    the cost function.

    NOTE: It eliminates the weights of least important features,
    therefore automatically performing feature selection and 
    producing a sparse model.

    @param X data array of feature values.
    @param y data array of labels.
    @return Fitted Lasso Regressor.
"""
def do_lasso_reg(X, y):
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    return lasso_reg


"""
    @brief Do Lasso Regression with Stochastic
    Gradient Descent.
    @param X data array of feature values.
    @param y data array of labels.
    @return Fitted SGDRegressor.
"""
def do_lasso_reg_with_sgd(X, y):
    # Specify pentaly as l1 to perform lasso reg.
    sgd_reg = SGDRegressor(penalty="l1")
    sgd_reg.fit(X, y.ravel())
    return sgd_reg