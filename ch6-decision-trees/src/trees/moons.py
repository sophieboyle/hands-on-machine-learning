from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


"""
    @brief Retrieves data from the moons dataset
    and divides them into training and testing sets.
    @return X_train Matrice of feature values for training
    @return X_test Matrice of feature values for testing
    @return y_train Array of labels
    @return y_test Array of labels
"""
def get_data():
    X, y = make_moons(n_samples=10000, noise=0.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


"""
    @brief Performs a grid search for a DecisionTreeClassifier
    to find the best hyperparameters on given training data.
    @param X_train feature value matrice for training.
    @param y_train array of labels.
    @return The best parameters.
"""
def find_best_param(X_train, y_train):
    tree_clf = DecisionTreeClassifier()
    param_grid = [
        {"max_leaf_nodes": [2, 3, 4, 5]},
    ]
    grid_search = GridSearchCV(tree_clf, param_grid)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_params_


"""
    @brief Trains the DecisionTreeClassifier on the full dataset.
    @param X_train Matrice of feature values for training.
    @param y_train Array of labels.
    @param max_leaf_nodes The hyperparameter for the model.
    @return The fitted DecisionTreeClassifier model.
"""
def train_best(X_train, y_train, max_leaf_nodes):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    tree_clf.fit(X_train, y_train)
    return tree_clf


"""
    @brief Test the given model on given testing data.
    @param X_test Matrice of feature values for testing.
    @param y_test Array of labels for testing.
    @param tree_clf Fitted DecisionTreeClassifier to test.
    @return Accuracy score.
"""
def test_model(X_test, y_test, tree_clf):
    y_pred = tree_clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    best_params = find_best_param(X_train, y_train)
    tree_clf = train_best(X_train, y_train, best_params["max_leaf_nodes"])
    acc_score = test_model(X_test, y_test, tree_clf)
    print(acc_score)