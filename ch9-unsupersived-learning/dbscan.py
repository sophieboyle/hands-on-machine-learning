from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.05)

    # DBSCAN works via basing clusters on core instances,
    # which are instances with at least min_samples
    # instances within its epsilon-neighbourhood.
    # These instances are grouped into the same cluster.

    # Instances without a core instance in its neighbourhood,
    # and that are not core instances themselves, are anomalies.
    # They will have the label: -1. 

    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X)

    # Returns the labels of all instances
    print(dbscan.labels_)
    # Returns the indices of core instances.
    print(dbscan.core_sample_indices_)
    # Returns the core instances.
    print(dbscan.components_)

    # The DBSCan algorithm cannot make predictions on new instances.
    # Instead it is possible to implement a predictor as follows:
    knn = KNeighborsClassifier(n_neighbors=50)
    # NOTE: We choose only to train on core instances here
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

    X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
    print(knn.predict(X_new))
    print(knn.predict_proba(X_new))

    # Seeing as there were no anomalies involved in training
    # the predictor, it is possible to introduce anomaly detection
    # by providing a maximum distance from any instances.
    y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
    y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
    y_pred[y_dist > 0.2] = -1
    print(y_pred.ravel())