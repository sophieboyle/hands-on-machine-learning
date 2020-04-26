import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


"""
    @brief Get iris array of feature values and labels.
    @return Matrice of feature values.
    @return Array of labels.
"""
def get_iris():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = (iris["target"] == 2).astype(np.float64)
    return X, y


"""
    @brief Fits a SVM classifier on given dataset X, with
    labels y.

    NOTE: Scaling of the data is performed in this function
    as SVMs are sensitive to feature scales. The hyperparam
    C is also set to 1, leaving a larger margin and therefore
    more margin violations - this ensures that it generalises
    better.

    @param X matrice of feature values.
    @param y array of labels.
    @return Fitted SVM model.
"""
def fit_svm(X, y):
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))
    ])
    svm_clf.fit(X, y)
    return svm_clf


"""
    @brief Alternatively train an SVM classifier by using
    the SGDClassifier class, which uses Stochastic Grad Desc.

    NOTE: While it will not converge as quickly, this is the
    better option for online classification or huge datasets.

    @param X matrice of feature values.
    @param y array of labels.
    @return Fitted SVM model.
"""
def fit_svm_with_grad_desc(X, y):
    sgd_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDClassifier(loss="hinge", alpha=1/(X.shape[0]*1)))
    ])
    sgd_clf.fit(X, y)
    return sgd_clf