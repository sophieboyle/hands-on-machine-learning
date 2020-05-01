from pre_proc import parse_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def show_digits(digits):
    for index, digits in enumerate(digits):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(digits.reshape(8, 8), cmap="binary", interpolation="bilinear")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Take 50 labelled instances
    X_train, X_test, y_train, y_test = parse_digits()
    n_labeled = 50
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
    print(X_train[:n_labeled])
    print(y_train[:n_labeled])

    # Having few labelled instances to train on results in
    # a low accuracy
    print(log_reg.score(X_test, y_test))

    # We can improve on this by using clustering
    k = 50
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Contains the distances from each digit (row) to 
    # the centroid of all clusters (column)
    X_digits_dist = kmeans.fit_transform(X_train)

    # For each cluster, find the digit closest to the centroid
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    X_representative_digits = X_train[representative_digit_idx]

    # We look at each representative digit and manually label them
    # show_digits(X_representative_digits)
    y_representative_digits = np.array([
                                        0, 1, 3, 2, 7, 6, 4, 6, 9, 5,
                                        1, 2, 9, 5, 2, 7, 8, 1, 8, 6,
                                        3, 2, 5, 4, 5, 4, 0, 3, 2, 6,
                                        1, 7, 7, 9, 1, 8, 6, 5, 4, 8,
                                        5, 3, 3, 6, 7, 9, 7, 8, 4, 9])
    
    # Therefore, instead of having 50 random labeled instances
    # we instead have instances which are representative of their cluster
    # This increases the accuracy.
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_representative_digits, y_representative_digits)
    print(log_reg.score(X_test, y_test)) 

    # It is possible to perform label propagation, where we
    # propagate the labels to all other instances in the cluster
    y_train_propagated = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i] 

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train_propagated)
    print(log_reg.score(X_test, y_test))

    # An even better idea is to only propagate the label to
    # the 20% of instances that are closest to the centroid
    percentile_closest = 20
    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

    for i in range(k):
        in_cluster = (kmeans.labels_ == i)
        cluster_dist = X_cluster_dist[in_cluster]
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (X_cluster_dist > cutoff_distance)
        X_cluster_dist[in_cluster & above_cutoff] = -1
    
    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train_propagated[partially_propagated]

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
    print(log_reg.score(X_test, y_test))