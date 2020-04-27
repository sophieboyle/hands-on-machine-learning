from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


"""
    @brief Retrieves the iris data, splitting it up
    into a matrice of feature values and an array of labels.
    @return X matrice of feature values.
    @return y array of labels.
    @return iris The full retrieved dataset.
"""
def get_iris():
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    return X, y, iris


"""
    @brief Fits a decision tree classifier on the given data.
    NOTE: Decision Trees don't require feature scaling/centering.
    @param X matrice of feature values
    @param y array of labels
    @return Fitted Decision tree classifier
"""
def fit_tree_clf(X, y):
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)
    return tree_clf


"""
    @brief Creates a dot file diagram of the given tree model.
    NOTE: Convert the dot file to png via
    dot -Tpng images/iris.dot -o images/iris.png
    @param Tree_clf The fitted tree model
    @param iris The full set of iris data
"""
def visualise(tree_clf, iris):
    export_graphviz(tree_clf, out_file="images/iris.dot",
                    feature_names=iris.feature_names[2:],
                    class_names=iris.target_names, rounded=True,
                    filled=True)


if __name__ == "__main__":
    X, y, iris = get_iris()
    tree_clf = fit_tree_clf(X, y)
    visualise(tree_clf, iris)
    # Return the probabilities for an iris with petal
    # length 5cm and width 1.5cm
    print(tree_clf.predict_proba([[5, 1.5]]))
    # The highest probability is at index 1 of the array
    # i.e. the iris versicolour class.
    print(tree_clf.predict([[5, 1.5]]))