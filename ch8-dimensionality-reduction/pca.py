import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


def manual_pca(X):
    """Produces a dataset with dimensionality reduced to
    2 dimension.

    Arguments:
        X {nparray} -- Dataset to reduce.

    Returns:
        nparray -- The dataset reduced to 2D.
    """
    # Centers the data about the origin.
    X_centered = X - X.mean(axis=0)
    # Uses singular value decomposition.
    U, s, Vt = np.linalg.svd(X_centered)
    # Retrieve the first two principal components.
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]
    
    # Retrieve the matrice W_d where d is the first d
    # columns of V, i.e. the dimensionality you wish
    # to reduce the data to. Here it is 2.
    W_2 = Vt.T[:, :2]
    # Therefore we get X_d_proj, which is the reduced dataset
    # with dimensionality d (here 2) obtained by projecting
    # the original dataset onto the hyperplane.
    X_2_proj = X_centered.dot(W_2)
    return X_2_proj


def auto_pca(X, dimensions):
    """Produces reduced dimensionality dataset
    using sklearn's inbuilt class.

    Arguments:
        X {nparray} -- Dataset to reduce.
        dimensions {int} -- Dimensions to reduce to.
    
    Returns:
        nparray -- The dataset reduced to the given dimension.
    """
    pca = PCA(n_components=dimensions)
    # Explained variance ratio will describe the proportion
    # of the dataset's variance which is maintained by each
    # principal component.
    print(pca.explained_variance_ratio_)
    X_d_proj = pca.fit_transform(X) 
    return X_d_proj


def find_dimensions(X):
    """Find the right number of dimensions to maintain
    at least 95% variance.

    NOTE: This is a manual approach, instead it is possible
    to simply perform PCA(n_components=0.95) to indicate the
    ratio of variance to conserve.
    
    You can also change svd_solver param to "full" to perform 
    full SVD instead of the default "auto" which uses 
    randomised SVD.

    Arguments:
        X {nparray} -- Dataset to investigate.

    Returns:
        int -- Number of dimensions maintaining significant
               variance.
    """
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    return d


def incremental_pca(X, n, d):
    """Performs incremental pca, which consists of splitting
    up the dataset into minibatches, and training the feeding
    the algorithm one batch at a time. This means that the whole
    training set does not need to fit into memory whilst fitting.

    Arguments:
        X {nparray} -- Dataset to reduce.
        n {int} -- Number of batches to split the dataset into.
        d {int} -- Number of dimensions to reduce the dataset to.

    Returns:
        nparray -- Dataset reduced to d dimensions.
    """
    inc_pca = IncrementalPCA(n_components=d)
    for X_batch in np.array_split(X, n):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    return X_reduced