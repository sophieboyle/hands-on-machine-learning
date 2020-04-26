from sklearn.svm import LinearSVR, SVR

"""
    @brief SVM can be used for regression as well.
    This involves fitting as many instances as possible
    within the margin (instead of outside, as with classification),
    and minimising the number of instances outside the margin.
    i.e. Margin violations in terms of SVM Regression regard
    instances found outside the margin instead of inside.

    NOTE: The Epsilon hyperparam determines the width
    of the margin.

    @param X matrice of feature values.
    @param y array of labels.
    @return Fitted SVM Regression model.
"""
def fit_svm_reg(X, y):
    svm_reg = LinearSVR(epsilon=1.5)
    svm_reg.fit(X, y)
    return svm_reg


"""
    @brief SVM regression may also be used with polynomial
    features using a kernel trick.
    @param X matrice of feature values.
    @param y array of labels.
    @return Fitted SVM Regression model with polynomial features.
"""
def fit_poly_svm_reg(X, y):
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)