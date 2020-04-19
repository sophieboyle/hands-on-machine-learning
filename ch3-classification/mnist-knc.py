from data import get
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

def main():
    mnist = get()

    # X.shape = (70000, 784), 70,000 isntances with 784 features
    # Each feature represents a pixel (28x28 img)
    # y.shape = (70000), contains labels for X
    X, y = mnist["data"], mnist["target"]
    
    # Seeing as the labels are currently strings,
    # we convert this to an integer 
    y = y.astype(np.uint8)

    # The mnist dataset is already segmented into
    # the first 60k instances being a training set
    # with the latter 10k being a test set 
    # NOTE: The training set has already been shuffled
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Initialise KNeighboursClassifier
    knc = KNeighborsClassifier()
    # knc.fit(X_train, y_train)

    # Do a grid search for the best hyperparams
    param_grid = [
        {"weights": ["uniform", "distance"], "n_neighbors":[2, 4, 6]},
    ]
    grid_search = GridSearchCV(knc, param_grid, scoring="neg_mean_squared_error",
                                return_train_score=True)
    grid_search.fit(X_train, y_train)

    print(f"BEST HYPERPARAMS: {grid_search.best_params_}")
    final_knc = grid_search.best_estimator_

    # Check the accuracy
    # score = cross_val_score(final_knc, X_train, y_train, cv=3, scoring="accuracy")
    # print(f"KNC VALIDATION ACCURACY: {score}")

    # Grid search prediction using the best estimator
    y_pred = grid_search.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()