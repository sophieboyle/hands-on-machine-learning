import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    # Momentum involves accelerating descent. The momentum hyperparam 
    # indicates friction, speeding up convergence.
    momentum_opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # Computes grad of cost function slightly ahead of the current location
    # in the direction of momentum. Converges faster. 
    nest_momentum_opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, 
                                            nesterov=True)
    
    # RMSProp variant of AdaGrad, which points the gradient closer
    # to the optimum ahead of time by decaying the learning rate
    # of steeper dimensions. RMSProp only uses gradients from
    # recent iterations. Rho indicates rate of decay.
    rmsprop_opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

    # Borrows techniques from both Momentum and RMSProp, by
    # tracking the exponentially decaying avg of prior grads, and 
    # also tracking exponentially decaying avg of prior squared grads.
    # There is also adjustment of momentum and squared grads away from
    # a bias towards 0 at the beginning of training (controlled by 
    # beta_1 and beta_2) 
    adam_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)