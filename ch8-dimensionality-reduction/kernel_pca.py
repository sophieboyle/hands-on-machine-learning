import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def do_kernel_pca(X):
    """Uses the kernel trick to perform kernelPCA with
    RBF kernel.

    Arguments:
        X {nparray} -- Dataset to reduce

    Returns:
        nparray -- Data reduced to 2 dimensions.
    """
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
    X_reduced = rbf_pca.fit_transform(X)
    return X_reduced


def log_reg_with_kpca(X, y):
    """Performs logistic regression on dimensionality-reduced
    data using a kernel PCA, and a grid-search to fine-tune
    the kernel pca's hyperparameters, finding the best gamma
    and kernel.

    Arguments:
        X {nparray} -- Matrice of feature values.
        y {nparray} -- Array of labels.

    Returns:
        GridSearchCV -- The result of the grid search.
    """
    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    
    return grid_search


def finding_reconstruction_error(X):
    """Finds the reconstruction error of a kernel PCA,
    i.e. the error in reconstructing the image from the
    reduced dataset produced by the kPCA.

    Arguments:
        X {nparray} -- Dataset to reduce (and reproduce).

    Returns:
        int -- Mean squared error of the original dataset against
        the reconstructed dataset.
    """
    # fit_inverse_transform=True automatically trains a supervised
    # learning algorithm, using projected instances as training data
    # and the original data as targets. The algorithm used is based on
    # kernel ridge regression.
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                        fit_inverse_transform=True)
    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)
    return mean_squared_error(X, X_preimage)