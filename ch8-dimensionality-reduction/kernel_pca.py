import numpy as np
from sklearn.decomposition import KernelPCA


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