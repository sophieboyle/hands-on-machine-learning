from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
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
    @brief Train an ensemble voting classifier based on
    three distinctive classifiers using hard voting.
    @param X_train matrice of feature values.
    @param y_train array of labels.
    @return List containing: Fitted Voting Classifier,
    LogisticRegressionClassifier, RandomForestClassifier
    and SupportVectorClassifier.
"""
def train_ensemble(X_train, y_train):
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()

    # voting="hard" means the predicted class is chosen based on
    # counting the number of predictions for each class among the
    # classifiers in the ensemble, and simply choosing the class
    # with the highest vote.
    voting_clf = VotingClassifier(
        estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
        voting="hard"
    )
    voting_clf.fit(X_train, y_train)
    
    return [voting_clf, log_clf, rnd_clf, svm_clf]


"""
    @brief Train an ensemble voting classifier based on
    three distinctive classifiers using soft voting.
    @param X_train matrice of feature values.
    @param y_train array of labels.
    @return List containing: Fitted Voting Classifier,
    LogisticRegressionClassifier, RandomForestClassifier
    and SupportVectorClassifier.
"""
def train_ensemble_soft(X_train, y_train):
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    # For soft voting, SVC should have a predict_proba method
    # This will unfortunately slow down training.
    svm_clf = SVC(probability=True)

    # voting="soft" means the predicted class is chosen based on
    # the class with the highest indicated probability, which is
    # averaged across all classifiers. NOTE: Requires all classifiers
    # to have a predict_proba method.
    voting_clf = VotingClassifier(
        estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
        voting="soft"
    )
    voting_clf.fit(X_train, y_train)
    
    return [voting_clf, log_clf, rnd_clf, svm_clf]


"""
    @brief Check the accuracy score on all given classifiers
    @param X_train Matrice of feature values for training
    @param y_train Array of labels for training
    @param X_test Matrice of feature values for testing
    @param y_test Array of labels for training
    @param classifiers List of classifiers
    @return A dictionary of the each classifier's accuracy score,
    keyed by the classifier's name.
"""
def check_accuracies(X_train, y_train, X_test, y_test, classifiers):
    accuracies = {}
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies[clf.__class__.__name__] = accuracy_score(y_test, y_pred)
    return accuracies


if __name__=="__main__":
    X_train, X_test, y_train, y_test = get_data()
    # classifiers = train_ensemble(X_train, y_train)
    classifiers = train_ensemble_soft(X_train, y_train)
    accuracies = check_accuracies(X_train, y_train, X_test, y_test, classifiers)
    for accuracy in accuracies.keys():
        print(str(accuracy) + ": " + str(accuracies[accuracy]))