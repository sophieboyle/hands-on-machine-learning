from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_iris_data():
    iris = datasets.load_iris()
    # Get just the petal widths
    X = iris["data"][:, 3:]
    # y = 1 if Iris Virginica, y = 0 otherwise
    y = (iris["target"] == 2).astype(np.int) 
    return X, y

def plot_iris_probas(X, y_proba):
    plt.plot(X, y_proba[:, 1], "g-", label="Iris Virginica")
    plt.plot(X, y_proba[:, 0], "b--", label="Not Iris Virginica")
    plt.show()

def iris_main():
    X, y = get_iris_data()

    # Train logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    
    # Plot probabilities of labels based on new data
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    plot_iris_probas(X_new, y_proba)

    # The graph indicates that there is a decision boundary
    # at 1.6cm width, indicative of the probabilities for 
    # either class being at 50%.
    print(log_reg.predict([[1.7], [1.5]]))