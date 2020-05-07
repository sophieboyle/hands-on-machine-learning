import tensorflow as tf

def huber_fn(y_true, y_pred):
    """Manual implementation of the huber loss function.
    NOTE: Delta is set to 1.

    Arguments:
        y_true {nparray} -- True labels
        y_pred {nparray} -- Predicted labels

    Returns:
        tensor -- Returns either the square loss or the linear
                  loss depending on whether the error is sufficiently
                  small.
    """
    error = y_true - y_pred
    is_small = tf.abs(error) < 1

    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5

    return tf.where(is_small, squared_loss, linear_loss)