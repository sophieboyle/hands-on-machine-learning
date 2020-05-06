import tensorflow as tf
from tensorflow import keras
import numpy as np


def exponential_decay(lr0, s):
    """Sets the exponential decay function according to
    given hyperparameters.

    Arguments:
        lr0 {float} -- Initial learning rate.
        s {float} -- Number of steps to drop the learning
                     rate in.
    
    Returns:
        function -- The exponential decay function initialised
                    with an initial learning rate and no of steps.
    """
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / 20)
    return exponential_decay_fn


def piecewise_constant(boundaries, values):
    """Sets the piecewise constant learning scheduling function
    according to a set of boundaries and values. This means that
    whenever an epoch crosses a boundary, a new learning rate value
    is used.

    Arguments:
        boundaries {list} -- List of epoch boundaries.
        values {list} -- List of learning rates.

    Returns:
        Function -- The learning schedule function.
    """
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn


if __name__ == "__main__":
    # Power Scheduling: Decreasing the learning rate
    # after each step by dividing the initial learning rate
    # by (1 + (t/s)^c) where t is the number of steps, and c 
    # is usually set to 1. s is the number of steps it takes
    # to divide the lr by another unit, and the decay hyperparam
    # is the inverse of s. 
    power_sch_opt = keras.optimizers.SGD(lr=0.91, decay=1e-4)

    # Getting an exponential decay function, which decreases
    # the learning rate exponentially after s number of steps.
    exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

    # Creating a callback with a schedule function
    # which updates the learning rate at each epoch. 
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

    # Performance scheduling decreases the learning rate by a factor
    # after no improvement has been shown for a given number of epochs.
    performance_lr_sch = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    