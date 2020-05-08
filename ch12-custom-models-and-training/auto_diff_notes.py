import tensorflow as tf
from tensorflow import keras


def f(w1, w2):
    # Basic function
    return 3 * w1**2 + 2 * w1 * w2


@tf.custom_gradient
def better_softplus(z):
    """Custom softplus with custom gradient function
    implementation, as autodiff cannot be used on softplus.
    Therefore it is possible to force tensorflow to use a 
    stable function instead.
    """
    exp = tf.exp(z)
    def better_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), better_softplus_gradients


if __name__ == "__main__":
    # Instead of computing each partial derivative (which would be difficult
    # for a DNN with many parameteres) an approximation can be found by
    # measuring the difference in the function's output when a param
    # is tweaked

    # Initialising
    w1, w2 = 5, 3
    eps = 1e-6

    # Computing partial derivatives
    df_by_dw1 = (f(w1 + eps, w2) - f(w1, w2)) / eps
    df_by_dw2 = (f(w1, w2 + eps) - f(w1, w2)) / eps

    print(df_by_dw1)
    print(df_by_dw2)

    # Autodiff replaces the need to call a function at least once
    # per param
    w1, w2 = tf.Variable(5.), tf.Variable(3.) 

    # Gradient tape records every op involving a tensor.variable
    # Persistent can be set if gradient method needs to be called
    # multiple times. Tape can be forced to watch any tensors too
    # with tape.watch().
    with tf.GradientTape() as tape:
        z = f(w1, w2)
    
    # Gradients can be computed of the result with regards to listed vars
    gradients = tape.gradient(z, [w1, w2])
    print(gradients)