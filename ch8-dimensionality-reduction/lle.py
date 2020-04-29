from sklearn.manifold import LocallyLinearEmbedding

def do_lle(X, d):
    """Performs dimensionality reduction on a given dataset
    using Locally Linear Embedding, which measures the linear
    relation of each instance to its closest neighbours,
    and finds a lower-dimensional representation in which most
    of these relationships are maintained.

    NOTE: LLE works particularly well on unrolling twisted
    manifolds.

    Arguments:
        X {nparray} -- Dataset to reduce.
        d {int} -- Number of dimensions to reduce the data to.

    Returns:
        nparray -- Reduced dataset.
    """
    lle = LocallyLinearEmbedding(n_components=d, n_neighbors=10)
    X_reduced = lle.fit_transform(X)
    return X_reduced