import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import LinearSVC


def parse_mnist():
    """ Gets and divides up the MNIST dataset into testing
        and training sets.

    Returns:
        nparray -- Matrice of feature values for training
        nparray -- Matrice of feature values for validation
        nparray -- Matrice of feature values for testing
        nparray -- Array of labels for training
        nparray -- Array of labels for validation
        nparray -- Array of labels for testing
    """
    mnist = data = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
                            X, y, test_size=10000, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
                            X_train_val, y_train_val, test_size=10000,
                            random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_estimators_separately(estimators, X_train, y_train, X_val, y_val):
    """Trains a list of estimators and returns a dictionary of their scores

    Arguments:
        estimators {List} -- List of estimators
        X_train {nparray} -- Matrice of feature values
        y_train {nparray} -- Array of labels
        X_val {nparray} -- Matrice of feature values
        y_val {nparray} -- Array of labels

    Returns:
        Dict -- Dictionary of estimator scores keyed by their class names.
    """
    scores = {}
    for estimator in estimators:
        estimator.fit(X_train, y_train)
        scores[estimator.__class__.__name__]  = estimator.score(X_val, y_val)
    return scores


def train_hard_voting(predictors, X_train, y_train):
    """Construct a voting ensemble classifier from a list of
    predictors.

    Arguments:
        predictors {List} -- List of predictor models
        X_train {nparray} -- Matrice of feature values
        y_train {nparray} -- Array of labels

    Returns:
        VotingClassifier -- Fitted voting classifier
    """
    est = [(predictor.__class__.__name__, predictor) for predictor in predictors]
    vot_clf = VotingClassifier(estimators=est, voting="hard")
    vot_clf.fit(X_train, y_train)
    return vot_clf


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = parse_mnist()

    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    ext_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    svm_clf = LinearSVC(random_state=42)

    scores = train_estimators_separately([rnd_clf, ext_clf, svm_clf],
                                        X_train, y_train, X_val, y_val)
    print(scores)

    vot_clf = train_hard_voting([rnd_clf, ext_clf, svm_clf],
                                X_train, y_train)
    print(vot_clf.score(X_val, y_val))

    # Removing the SVM Classifier due to its low performance
    del vot_clf.estimators_[2]
    print(vot_clf.score(X_val, y_val))

    # Try with soft-voting
    vot_clf.voting = "soft"
    print(vot_clf.score(X_val, y_val))

    # Switch back to hard-voting
    # vot_clf.voting = "hard"

    # Get results on the testing set
    print(vot_clf.score(X_test, y_test))
    print([estimator.score(X_test, y_test) for estimator in vot_clf.estimators_])
