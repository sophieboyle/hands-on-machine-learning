from data import get, view_example
from helper import plot_precision_recall_vs_threshold
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve

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

    

if __name__=="__main__":
    main()