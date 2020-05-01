from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def parse_digits():
    X_digits, y_digits = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
    return X_train, X_test, y_train, y_test


def train_log_reg(X_train, y_train):
    log_reg = LogisticRegression(max_iter=50000)
    log_reg.fit(X_train, y_train)
    return log_reg


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = parse_digits()
    log_reg = train_log_reg(X_train, y_train)
    # Basic log_reg score on data is 96% accuracy
    print(log_reg.score(X_test, y_test))

    # Using K-Means as preprocessing, we can perform
    # dimensionality reduction to the dataset as preproc
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=50)),
        ("log_reg", LogisticRegression(max_iter=50000)),
    ])
    pipeline.fit(X_train, y_train)
    # The accuracy with K-Means as preproc increased to 98%
    print(pipeline.score(X_test, y_test))

    # A grid search can also be used to find the best
    # number of clusters
    param_grid = dict(kmeans__n_clusters=range(2, 100))
    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
    grid_clf.fit(X_train, y_train)
    print(grid_clf.best_params_)
    print(grid_clf.score(X_test, y_test))    