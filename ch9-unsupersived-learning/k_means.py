from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np


def train_k_means(k, X):
    """Train a K Means clusterer on the given data,
    finds each cluster's center and assigns every instance
    to its closest cluster.

    This is HARD clustering, and only takes into account
    the distance from the instance to the centroid when
    assigning instances to clusters, therefore it performs
    badly on clusters with varying diameter.
    
    NOTE: The number of clusters to be found must be specified.
    The data must also be scaled before using any K-Means alg,
    as otherwise, clusters may be stretched.

    Arguments:
        k {int} -- Number of clusters to find.
        X {nparray} -- Dataset to fit.
    
    Returns:
        nparray -- The predictions for the clusters.
        KMeans -- The fitted K-Means clustering algorithm.
    """
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    return y_pred, kmeans


def train_mini_batch_k_means(k, X):
    """Trains a variant of the K-Means algorithm, which 
    trains on mini-batches of the full dataset upon each
    iteration.
    
    NOTE: This is faster and useful if a huge dataset cannot
    fit in memory. However, the intertia is slightly worse,
    and grows with the number of clusters.

    Arguments:
        k {int} -- Number of clusters
        X {nparray} -- Dataset

    Returns:
        MiniBatchKMeans -- A fitted mini-batch k-means alg.
    """
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k)
    minibatch_kmeans.fit(X)
    return minibatch_kmeans


def gen_blobs():
    """Generates data which appears in the form of blobs
    to be used as an example for k-means clustering.

    NOTE: The values in the y array represent the index of
    the cluster an index is to be found in. (They are NOT labels
    in the classification sense.)

    Returns:
        nparray -- Dataset
        nparray -- Cluster index array
    """
    blob_centers = np.array(
                            [[ 0.2,  2.3],
                             [-1.5 ,  2.3],
                             [-2.8,  1.8],
                             [-2.8,  2.8],
                             [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                    cluster_std=blob_std, random_state=7)
    return X, y


if __name__ == "__main__":
    X, y = gen_blobs()
    y_pred, kmeans = train_k_means(5, X)
    print(y_pred)
    print(kmeans.labels_)

    # It is possible to view the centroids found by the algorithm
    print(kmeans.cluster_centers_)

    # Assigning new instances to the cluster with the closest centroid
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    print(kmeans.predict(X_new))

    # Transforming returns the distance between each instance (row)
    # to every centroid (column)
    print(kmeans.transform(X_new))

    # Since centroids are initialised randomly, (it is actually
    # initialised using K-Means++ which tries to select distant
    # centroids in scikit-learn), it is possible for the k-means 
    # clusterer to get stuck in a local optimum.
    # The likeliness of this can be mitigated by centroid initialisation

    # If a general idea of where centroids should be are known, it 
    # is possible to use this approximation for initialisation.
    good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
    kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
    # NOTE: n_init also makes it possible to initialise the clusterer
    # several times with different initialisations.

    # Returns the performance metric inertia, which is the mean
    # squared dist between each instance and its closest centroid.
    print(kmeans.inertia)
    # Score returns negative intertia
    print(kmeans.score(X))

    # The silhouette score is the mean silhouette coefficient of
    # all instances, which is (b-a)/max(a,b), where a is the
    # intra-cluster distance and b is the mean nearest-cluster dist.
    # The silhouette coeff. ranges between -1 and +1, where:
    # +1 indicates an instance is well within its own cluster
    # -1 indicates an instance may have been allocated to the wrong cluster
    print(silhouette_score(X, kmeans.labels_))

    # NOTE: Silhouette score is much better when determining how
    # many clusters to allocate to an algorithm, as generally
    # the inertia will decrease with the number of clusters anyway,
    # but this may not actually result in the desired effect.
    # Silhouette score however is much more detailed in identifying
    # how instances relate to their (and other) clusters as a metric.  