import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time


def parse_mnist():
    """ Gets and divides up the MNIST dataset into testing
        and training sets.
    Returns:
        nparray -- Matrice of feature values for training
        nparray -- Matrice of feature values for testing
        nparray -- Array of labels for training
        nparray -- Array of labels for testing
    """
    mnist = data = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    
    X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=10000, random_state=42)

    return X_train, X_test, y_train, y_test


def time_training(model, X, y):
    """Times the model whilst training.

    Arguments:
        model {Model} -- Model to train.
        X_train {nparray} -- Matrice of feature values.
        y_train {nparray} -- Array of labels.

    Returns:
        float -- Time taken for the model to train.
        model -- The fitted model.
    """
    start = time.time()
    model.fit(X, y)
    result = time.time() - start
    return result, model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = parse_mnist()
    # Time and train a normal random forest classifier on mnist
    rnd_clf = RandomForestClassifier(random_state=42)
    rnd_time, rnd_clf = time_training(rnd_clf, X_train, y_train)
    print(f"Time taken to train: {rnd_time}")

    # Check accuracy of normal random forest classifier
    y_pred = rnd_clf.predict(X_test)
    print(f"Test score: {accuracy_score(y_test, y_pred)}")

    # Reduce the mnist dataset
    pca = PCA(n_components=0.95)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    # Time the training on new classifier with reduced data
    rnd_clf_v2 = RandomForestClassifier(random_state=42)
    rnd_time_v2, rnd_clf_v2 = time_training(rnd_clf_v2, X_train_reduced, y_train)
    print(f"Time taken to train reduced: {rnd_time_v2}")

    # Check accuracy of random forest classifier on reduced data
    y_pred_v2 = rnd_clf_v2.predict(X_test_reduced)
    print(f"Test score reduced: {accuracy_score(y_test, y_pred_v2)}")

    # Now try on softmax regression
    log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", 
                                random_state=42)
    log_time, log_reg = time_training(log_reg, X_train, y_train)
    print(f"Time taken to train logreg: {log_time}")
    y_pred_log = log_reg.predict(X_test)
    print(f"Test score logreg: {accuracy_score(y_test, y_pred_log)}")

    # And also softmax regression on the reduced dataset
    log_reg_v2 = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                    random_state=42)
    log_time_v2, log_reg_v2 = time_training(log_reg_v2, X_train_reduced, y_train)
    print(f"Time taken to train reduced: {log_time_v2}")
    y_pred_log_v2 = log_reg_v2.predict(X_test_reduced)
    print(f"Test score reduced: {accuracy_score(y_test, y_pred_log_v2)}")

    # NOTE: PCA can but does not always improve training time.