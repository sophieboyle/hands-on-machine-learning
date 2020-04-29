from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from moons import *


"""
    @brief Simply train a decision tree classifier.
    @param X_train matrice of feature values.
    @param y_train array of labels.
"""
def train_tree(X_train, y_train):
    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(X_train, y_train)
    return tree_reg


"""
    @brief Returns the model's residual error.
    @param tree_reg Fitted model
    @param X matrice of feature values
    @param y array of labels
    @return The residual errors.
"""
def get_residual_error(tree_reg, X, y):
    return y - tree_reg.predict(X)


"""
    @brief Manually perform gradient boosting with
    a given number of predictors. Trains 3 trees which
    each attempt to fit the residual errors of their 
    predecessor.
    @param count T The number of predictors.
    @param X matrice of feature values.
    @param y array of labels.
    @return A list of the predictors constructing the
    ensemble.
"""
def manual_boost(count, X, y):
    # Contains the models in the ensemble
    ensemble = []
    # Initialise the residual error
    res_err = y

    for i in range(count):
        # Train a tree
        tree_reg = train_tree(X, res_err)
        # Add trained tree to the ensemble
        ensemble.append(tree_reg)
        # Obtain the residual errors of the tree for the next
        res_err = get_residual_error(tree_reg, X, y)
    
    return ensemble


"""
    @brief Return a prediction for the ensemble by summing
    the predictions of each tree in the ensemble.
    @param ensemble List of predictors.
    @param X_new The new data to make a prediction upon.
    @return The prediction given by the ensemble.
"""
def get_ensemble_predictions(ensemble, X_new):
    y_pred = sum(tree.predict(X_new) for tree in ensemble)
    return y_pred


"""
    @brief Does the same as above, training 3 predictors
    using gradient boosting.
    @param X_train Matrice of feature values.
    @param y_train Array of labels.
    @return Fitted GradientBoostingRegressor.
"""
def train_boost(X_train, y_train):
    # Learning rate controls the contribution of each tree
    # in the ensemble.
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                    learning_rate=1.0)
    gbrt.fit(X_train, y_train)
    return gbrt


"""
    @brief Use early stopping to find the best number
    of estimators for the a GradientBoostingRegressor.

    NOTE: The implementation of "early stopping" for this
    algorithm involves training all 120 estimators, and then
    post-training, going back to see which amount of 
    estimators produces the best result.

    @param X matrice of feature values.
    @param y array of labels
    @return A fitted GradientBoostingRegressor.
"""
def grad_boost_fake_early_stop(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    grbt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    grbt.fit(X_train, y_train)

    # Find the best number of estimators by calculating
    # the mean squared error of the ensemble at each
    # sequential training stage.
    errors = [mean_squared_error(y_val, y_pred) 
                for y_pred in grbt.staged_predict(X_val)]
    print(errors)
    best_n_estimators = np.argmin(errors) + 1

    # Train new GradientBoostingRegressor on best estimator hyperparam
    gbrt_best = GradientBoostingRegressor(max_depth=2, 
                                        n_estimators=best_n_estimators)
    gbrt_best.fit(X_train, y_train)
    return gbrt_best


"""
    @brief Perform early stopping on a gradient boosting
    regressor.
    @param X matrice of feature values.
    @param y array of labels.
    @param n Max number of estimators to attempt to train.
    @param max_error The maximum number of times to allow
    for the error to increase before stopping.
    @return The final best estimator.
"""
def grad_boost_real_early_stop(X, y, n, max_error):
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    # Maintains the minimum error seen
    min_val_error = float("inf")
    # Counts how many times the error has sequentially risen
    error_going_up = 0
    # warm_start=True keeps existing trees when calling fit()
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
    # Saves the best model
    best_model = gbrt

    for n_est in range(1, n+1):
        # Set the number of estimators
        gbrt.n_estimators = n_est
        # Fit and find the validation error for this ensemble
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_err = mean_squared_error(y_val, y_pred)

        # Comparing the new validation error to the current minimum
        if val_err < min_val_error:
            min_val_error = val_err
            error_going_up = 0
            best_model = gbrt
        else:
            error_going_up += 1
            # Stop training if the error keeps rising
            if error_going_up == max_error:
                break
    
    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    ensemble = manual_boost(3, X_train, y_train)

    best_grad_boost = grad_boost_fake_early_stop(X_train, y_train)

    # NOTE: xgboost is a good lib, which provides an optimised
    # implementation of gradient boosting.