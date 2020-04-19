from data import get, view_example
from helper import plot_precision_recall_vs_threshold, plot_roc_curve, plot_digit
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
    #plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

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
                                        y_train_5, cv=3,
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
    #plt.show()
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

    # NOTE: Usually we should explore data preparation options, 
    # different models, fine-tuning hyperparams, and automation
    # For the sake of the exercise we assume that a good model has been found
    
    # We look at the confusion matrix of sgd_clf with scaled inputs
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    # Instead of purely viewing the numbers, we can show the matrix
    # in image form
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    # If most images are along the main diagonal, this shows correct
    # classification. We can see which particular classes are correctly
    # classified less often if they are darker than the others. It may
    # also be the case that there are not enough of that class in the data
    #plt.show()

    # To get a representation of error rates, we divide each of the
    # conf matrix's values by the number of instances in the class
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    # Diagonal is filled with zeros to keep only errors
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    #plt.show()

    # Multilabel classification is when an instance can belong
    # to multiple classes
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    # We train a KNeighboursClassifier on the multilabel instances
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    # A prediction on the KNeighboursClassifier will now output two labels
    print(f"KNEIGHBOURS MULTILABEL PREDICTION: {knn_clf.predict([X[0]])}")

    # To evaluate, we can compute the F1 score for each label,
    # and compute the average
    # NOTE: This metric assumes all labels are equally important
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    knn_clf_f1_score = f1_score(y_multilabel, y_train_knn_pred, average="macro")
    print(f"KNEIGHBOURS MULTILABEL F1 SCORE: {knn_clf_f1_score}")
    # NOTE: If we want to change the computation of the F1 score to place
    # more weight upon instances which are more supported, 
    # we change average="weighted"

    # Multioutput classification is when a label is not a boolean
    # i.e. a label can have more than 2 values

    # To indicate an example, we set up a system removing noise from images
    # This outputs an array of pixel intensities
    # The classifier's output has one label per pixel (multilabel)
    # and also each label's value can range from 0-255 (multioutput)

    # Add noise to the test and train sets, and set the original
    # images to the labels (y's)
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test

    # Train the KNeighboursClassifier to clean the image
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[X[0]]])
    plot_digit(clean_digit)

if __name__=="__main__":
    main()