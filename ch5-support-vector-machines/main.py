from src.soft_margin import *

def main():
    X, y = get_iris()
    svm_clf = fit_svm(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))

    sgd_clf = fit_svm_with_grad_desc(X, y)
    print(sgd_clf.predict([[5.5, 1.7]]))

if __name__=="__main__":
    main()