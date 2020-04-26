from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC


"""
    @brief Use SVM Classifier to fit data by first
    adding polynomial features, followed by scaling
    of the data.
    @param X matrice of feature values
    @param y array of labels
    @return Fitted SVM model
"""
def fit_poly_svm(X, y):
    poly_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])
    poly_svm_clf.fit(X, y)
    return poly_svm_clf


"""
    @brief Fit a SVM model using the kernel trick, which
    gets a result as if many polynomial features had been
    added, without actually adding many new features.
    @param X matrice of feature values
    @param y array of labels
    @return Fitted SVM model
"""
def fit_poly_kernel_svm(X, y):
    # coef0 controls how much the model is influenced by
    # high deg. polynomials vs low deg. polynomials.
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)
    return poly_kernel_svm_clf


"""
    @brief Uses simularity features by adding new features
    according to the simularity function: Guassian Radial Basis.
    Works via adding the simularity features and dropping
    the originals. Using the kernel trick, we can do this
    computation without actually adding many new similarity features.
    @param X matrice of feature values
    @param y array of labels
    @return Fitted SVM model
"""
def fit_rbf_kernel_svm(X, y):
    # Gamma determines the shape of the bell curve (i.e.
    # larger gamma, narrower bell curve. This leads to a
    # more irregular decision boundary.)
    # NOTE: If the model is overfitting, reduce gamma
    # If it is underfitting, increase gamma.
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
    rbf_kernel_svm_clf.fit(X, y)
    return rbf_kernel_svm_clf