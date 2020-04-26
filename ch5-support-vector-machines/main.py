from src.soft_margin import *
from src.non_lin import *
from sklearn.datasets import make_moons

def test_soft_margin():
    X, y = get_iris()
    svm_clf = fit_svm(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))

    sgd_clf = fit_svm_with_grad_desc(X, y)
    print(sgd_clf.predict([[5.5, 1.7]]))

def test_non_lin():
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    # NOTE: Currently gives convergence warning
    poly_svm_clf = fit_poly_svm(X, y)
    
def main():
    test_non_lin()

if __name__=="__main__":
    main()