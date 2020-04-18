from data import get, view_example
from helper import plot_precision_recall_vs_threshold, plot_roc_curve
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt

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

    # Binary classifiers can detect whether an instance is
    # one thing, or is not that one thing
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    # SGD classifier handles large datasets efficiently
    # as it deals with instances independently
    sgd_clf = SGDClassifier(random_state=42)
    # We train the classifier to detect whether or not
    # an instance is a number 5.
    sgd_clf.fit(X_train, y_train_5)
    
    # Evaluating the performance using 3 folds
    # Cross validation is not useful in this situation, as a small
    # percentage of instances will be a 5 anyway
    print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

    # Instead, we construct a confusion matrix to check how many
    # times the classifier confused one thing for another
    # For this we get a prediction set using 3 folds
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    conf_matrix = confusion_matrix(y_train_5, y_train_pred)
    print(conf_matrix)

    # Other metrics to look at would be:
    # Precision: How many detected positives were True Positives (TP/(TP+FP))
    # Recall: How many actual positives were True positives (TP/(TP+FN))
    print(f"PRECISION SCORE: {precision_score(y_train_5, y_train_pred)}")
    print(f"RECALL SCORE: {recall_score(y_train_5, y_train_pred)}")

    # We can look at the F1 score, which combines precision and recall
    # As it is the harmonic mean, it will only have a high score
    # if both the precision and recall are high.
    # NOTE: This favours classifiers with similar prediction and recall.
    print(f"F1 SCORE: {f1_score(y_train_5, y_train_pred)}")

    # As the classifier chooses whether an instance is pos or neg
    # depending on whether a score is above or below a threshold
    # We can look at the scores (instead of predictions) and then
    # Determine our own threshold for predictions
    y_example_scores = sgd_clf.decision_function([X[0]])
    print(f"CLF SCORE FOR X[0]: {y_example_scores}")
    threshold = 0
    y_example_pred = (y_example_scores > threshold)
    print(f"AT THRESHOLD 0: {y_example_pred}")
    threshold = 8000
    y_example_pred = (y_example_scores > threshold)
    print(f"AT THRESHOLD 8000: {y_example_pred}")

    # To decide which threshold to use, we use cross_val_predict
    # to instead give us the scores instead of the predictions
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                method="decision_function")
    # We can now compute the precision and recall values for
    # all possible thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    # We use the lowest threshold which gives 90% precision
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    # Predictions can now be made by comparing the classifier scores
    # to our selected threshold
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    print("PRECISION AND RECALL SCORES USING THRESHOLD 90:")
    print(precision_score(y_train_5, y_train_pred_90))
    print(recall_score(y_train_5, y_train_pred_90))

    # The ROC CURVE plots sensitivity (recall) verus 1 - specificity
    # Where Sensitivity is the True Positive Rate (TPR)
    # Specificity is the True Negative Rate (TNR)
    # Therefore the False Positive Rate (FPR) = 1 - TNR
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    plot_roc_curve(fpr, tpr)
    # We can find the area under the curve
    # Where a perfect classifier has AUC = 1
    # And a purely random classifier has AUC = 0.5
    print(f"ROC AUC SCORE: {roc_auc_score(y_train_5, y_scores)}")

    # Choose the PR Curve over the ROC curve only when the
    # positive class is rare, or when you care more about
    # minimising false positives over false negatives.
    # NOTE: In this case, since the positive class (getting a 5)
    # is rare, we opt to refer to the PR Curve.

    # We will now train a RandomForestClassifier and compare
    # ROC Curves and ROC AUC scores.
    forest_clf = RandomForestClassifier(random_state=42)
    # Seeing as there is no decision_function for RFC,
    # we use predict_proba which gives an array containing probabilities
    # as to whether an instance belongs to a given class
    y_probas_forest = cross_val_predict(forest_clf, X_train,
                                        y_train, cv=3,
                                        method="predict_proba")

    # We get the probabilities of getting the positive class
    y_scores_forest = y_probas_forest[:, 1]
    # We can give the roc_curve probabilities instead of scores
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    # Plotting the SGD and Random Forest together show that
    # the Random Forest performs much better than the SGD
    plt.plot(fpr, tpr, "b:", label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right")
    plt.show()
    # On top of that, the Random Forest's ROC AUC is much better
    print(f"RANDOMFORESTCLASSIFIER'S ROC AUC: {roc_auc_score(y_train_5, y_scores_forest)}")

    # Use a SupportVectorMachine for multiclass classification
    # Scikit-learn automatically chooses whether to use
    # OvR or OVO (In this case it uses OvO)
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    print(svm_clf.predict([X[0]]))
    # The decision function shows that 10 scores are returned
    # per instance, indicating the score for each possible class
    print(f"MULTICLASS SVC SCORES ON X[0]: {svm_clf.decision_function([X[0]])}")

    # We can manually force scikit-learn to use OvR or OvO
    ovr_clf = OneVsRestClassifier(SVC())
    ovr_clf.fit(X_train, y_train)
    print(f"OVR CLASSIFIER PREDICTION ON X[0]: {ovr_clf.predict([X[0]])}")

    # We can train an SGDClassifier for Multiclass Classification
    # NOTE: SGD does not use OvR/OvO, as it can directly classify instances
    # into multiple classes.
    sgd_clf.fit(X_train, y_train)
    print(f"SGD_CLF MULTICLASS PREDICTION ON X[0]: {sgd_clf.predict([X[0]])}")

    # Evaluating the SGD multiclass classifier using cross validation
    sgd_clf_cv_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    print(f"CROSS VAL SCORE ON SGD_CLF: {sgd_clf_cv_score}")

    # Scaling the inputs increases accuracy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    sgd_clf_cv_score_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    print(f"CROSS VAL SCORE ON SGD_CLF SCALED INPUT: {sgd_clf_cv_score}")

if __name__=="__main__":
    main()