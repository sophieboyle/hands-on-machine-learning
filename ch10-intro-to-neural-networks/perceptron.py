import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


def get_iris():
    iris = load_iris
    X = iris.data[:, (2, 3)] # Gets petal length and width
    y = (iris.target == 0).astype(np.int) # Checks if iris setosa
    return X, y


if __name__ == "__main__":
    X, y = get_iris()

    # Perceptrons are ANN architectures based off of TLUs
    # where all input connections are associated with a
    # specific weight. This weight is adjusted by reinforcing
    # the connections which contribute to the correct prediction.

    # The TLU outputs the result of a step function applied to
    # the weighted sum of its inputs.  

    # The Perceptron is a single layer of TLUs. Inputs are fed
    # through the input layer, whilst there is an additional
    # bias neuron which outputs 1 all the time (x_0 = 1)

    # The result across several instances is computed via:
    # h(X) = fi(XW + b), where X is the dataset, W is the weight
    # matrix, b is the bias vector (one bias term per neuron),
    # and fi is the activation function. (Step function for TLUs)

    per_clf = Perceptron()
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)