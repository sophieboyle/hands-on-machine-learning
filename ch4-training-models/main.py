from closed_form_sol import *

def main():
    X, y = gen_lin_data()
    theta = compute_best_model_param(X, y)
    pred, X_fit = ex_predict_using_theta(theta)
    plot_predictions(X, y, X_fit, pred)

if __name__=="__main__":
    main()