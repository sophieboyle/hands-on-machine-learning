import numpy as np


"""
    @brief Given a learning rate and some data,
    compute the best model parameter using batch
    gradient descent.
    
    NOTE: As batch grad desc, trains
    on the whole training set to allow for a steady,
    regular decrease towards the optimal solution,
    it does not perform well on large training sets.

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


"""
    @brief Use learning schedule to gradually reduce
    the learning rate of an algorithm.
    @param t0 Numerical learning schedule hyperparam
    @param t1 Numerical learning schedule hyperparam
    @param t current learning rate
    @return New learning rate
"""
def learning_schedule(t0, t1, t):
    return t0 / (t + t1)


"""
    @brief Using Stochastic Gradient Descent, compute
    the best model parameter. Uses a learning schedule
    to adjust the learning rate at every iteration, to
    allow for settling near the minimum.

    NOTE: This performs better than batch gradient descent
    on large training sets as a result of only using one
    instance per iteration for training. However, this makes
    the descent more randomised, which allows for it to escape
    local minima, but does not however allow it to naturally
    settle near the global minima - this is why we use the
    learning schedule.

    NOTE: It is also important to assume that X_b and y are
    independent and identically distributed (make sure to 
    shuffle the data respectively before use).

    @param X_b array of data with x0 added.
    @param y array of labels
    @param m Number of instances 
    @param theta=np.random.randn(2,1) Random initialisation
    @param t0=5 Hyperparameter to the learning schedule
    @param t1=50 Hyperparameter to the learning schedule
    @param n_epochs Number of iterations
    @return The best model parameter found.
"""
def stochastic_grad_desc(X_b, y, m, theta=np.random.randn(2,1), t0=5, 
                        t1=50, n_epochs=50):
    for epoch in range(n_epochs):
        for i in range(m):
            # Randomly select an instance to compute on
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Compute the gradient vector of xi and yi
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)

            # Adjust the learning rate according to the
            # learning schedule function
            n = learning_schedule(t0, t1, epoch * m + i)
            # Find next step
            theta = theta - n * gradients
    
    return theta