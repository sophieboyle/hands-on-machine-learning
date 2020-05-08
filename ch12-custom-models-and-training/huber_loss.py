import tensorflow as tf
from tensorflow import keras


def create_huber(threshold=1.0):
    """Generates a huber loss function according to a given
    threshold

    Keyword Arguments:
        threshold {float} -- The delta at which to consider
                             an error sufficiently small
                             (default: {1.0})
    """ 
    def huber_fn(y_true, y_pred):
        """Manual implementation of the huber loss function.

        Arguments:
            y_true {nparray} -- True labels
            y_pred {nparray} -- Predicted labels

        Returns:
            tensor -- Returns either the square loss or the linear
                      loss depending on whether the error is sufficiently
                      small.
        """
        error = y_true - y_pred
        is_small = tf.abs(error) < threshold

        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2

        return tf.where(is_small, squared_loss, linear_loss)
    return huber_fn


def load_huber_model(filename, threshold):
    """Returns a model loaded which was compiled using a huber loss
    function.

    Arguments:
        filename {string} -- .h5 model file
        threshold {float} -- Threshold of the huber loss function.

    Returns:
        Keras Model -- Loaded model.
    """
    return keras.models.load_model(filename, 
            custom_objects={"huber_fn" : create_huber(threshold)})


class HuberLoss(keras.losses.Loss):
    # By specifying HuberLoss as a class which subclasses keras'
    # Loss class, it is possible to avoid having to hardcode the
    # threshold when loading a saved model.
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
    
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < threshold

        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2

        return tf.where(is_small, squared_loss, linear_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


class HuberMetric(keras.metrics.Metric):
    """Huber Loss as a streaming metric (one which is gradually
    updated upon each batch.) Streaming metrics are useful for
    metrics which cannot simply be computed over the mean.
    NOTE: This is not the case for Huber Loss, this is simply
    an example.

    Arguments:
        keras {Metric} -- keras.metrics.Metric
    """
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)

        # Tracks metric's state over batches
        # Sum of all huber losses
        self.total = self.add_weight("total", initializer="zeros")
        # All instances seen thus-far
        self.count = self.add_weight("count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Called when object is called as function.
        # Computes metric
        metric = self.huber_fn(y_true, y_pred)
        # Appends to the total huber losses
        self.total.assign_add(tf.reduce_sum(metric))
        # Adds the batch of instances to the count
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        # Computes final metric result: huber metric across all instances
        return self.total / self.count

    def get_config(self):
        # Ensures threshold is saved when saving a model
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}