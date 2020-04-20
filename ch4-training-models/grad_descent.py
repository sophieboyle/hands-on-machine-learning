import numpy as np


"""
    @brief Given a learning rate and some data,
    compute the best model parameter using batch
    gradient descent.
    @param n a floating point learning rate
    @param X data to compute best model param for
    @param y labels to compute best model param for
    @param number of instances
    @return theta array of numerical best model parameters
"""
def batch_grad_desc_best_theta(n, X, y, m):
    n_iterations = 1000
    # Random step initialisation
    theta = np.random.randn(2, 1)
    
    for iterations in range(n_iterations):
        # Compute gradient vector
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        # Compute the next step
        theta = theta - n * gradients
    
    return theta