from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs
import numpy as np


def get_data():
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]
    return X, y


if __name__ == "__main__":
    X, y = get_data()

    # NOTE: Default number of initialisations is 1
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    # Vector of weights for each cluster
    print(f"Weights: {gm.weights_}")

    # Means deciding where the instance is sampled within
    # the Gaussian Distribution (a cluster represents a single
    # Guassian Distr.)
    print(f"Means:\n {gm.means_}")

    # Covariance matrice which also decides where the instance
    # is sampled within the Gaussian Distribution. 
    print(f"Covariance matrices:\n {gm.covariances_}")

    print(gm.converged_)
    print(gm.n_iter_)

    # New instances can be assigned via either hard or soft clustering
    print(gm.predict(X))
    # Soft clustering uses probabilities
    print(gm.predict_proba(X))

    # Seeing as Guassian Mixture is a generative model, it is possible
    # to sample new instances
    X_new, y_new = gm.sample(6)
    print(X_new)
    print(y_new)

    # Estimating the log of the probability density function for 
    # each instance
    print(gm.score_samples(X))

    # Gaussian Mixture models can be used for anomaly detection
    # According to a density-threshold, instances found in
    # low density regions can be classed as anomalies 
    densities = gm.score_samples(X)
    density_threshold = np.percentile(densities, 4)
    anomalies = X[densities < density_threshold]

    # To measure the appropriate number of clusters to use
    # we can use theoretical information criterion BIC and AIC
    # Both use the maximised value of the likelihood function.
    # Select the model with the lowest BIC/AIC 
    print(gm.bic(X))
    print(gm.aic(X))

    # BayesianGaussianMixture gives a weight of 0 to unnecessary
    # clusters, therefore automatically detecting how many clusters
    # are actually needed.  
    bgm = BayesianGaussianMixture(n_components=10, n_init=10)
    bgm.fit(X)
    # The alg. shows that only 3 out of the 10 clusters are required.
    print(np.round(bgm.weights_, 2))