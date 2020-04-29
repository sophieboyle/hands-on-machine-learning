from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from moons import *


"""
    @brief Uses bagging to train an ensemble
    of 500 decisionTreeClassifiers, each on
    100 training instances.

    NOTE: It is also 
    
    @param X_train matrice of feature values.
    @param y_train array of labels.
    @return Fitted bagging ensemble model.
"""
def train_bagging(X_train, y_train):
    # bootstrap=True indicates bagging
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1,
        oob_score=True
    )
    bag_clf.fit(X_train, y_train)
    return bag_clf


"""
    @brief Uses pasting to train an ensemble
    of 500 decisionTreeClassifiers, each on
    100 training instances.
    @param X_train matrice of feature values.
    @param y_train array of labels.
    @return Fitted bagging ensemble model.
"""
def train_pasting(X_train, y_train):
    # bootstrap=False indicates bagging
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=False, n_jobs=-1
    )
    bag_clf.fit(X_train, y_train)
    return bag_clf


"""
    @brief Training a randomForestClassifier.
    
    NOTE: Acts in the same way that a baggingClassifier
    on multiple DecisionTreeClassifiers works, however
    the RandomForestClassifier is OPTIMISED for this.

    @param X_train matrice of feature values.
    @param y_train array of labels.
"""
def train_rand_for(X_train, y_train):
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                    n_jobs=-1)
    rnd_clf.fit(X_train, y_train)
    return rnd_clf


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    bag_clf = train_bagging(X_train, y_train)
    rnd_clf = train_rand_for(X_train, y_train)

    # OOB SCORE is the avg score of evaluations on out of bag
    # instances (i.e. the instances which each predictor
    # has not seen during training - which is different for each)
    print(bag_clf.oob_score_)
    # The above OOB SCORE indicates what the expected accuracy on
    # the test set will be
    print(check_accuracies(X_train, y_train, X_test, y_test,
                            [bag_clf, rnd_clf]))
