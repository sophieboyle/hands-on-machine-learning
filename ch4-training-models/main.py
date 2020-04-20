from lin_reg.closed_form_sol import *
from lin_reg.lin_reg import *
from lin_reg.grad_descent import *
from polynomial_reg.polynomial_reg import *


def test_lin_reg():
    X, y = gen_lin_data()
    theta = compute_best_model_param(X, y)

    pred, X_fit = ex_predict_using_theta(theta)
    plot_predictions(X, y, X_fit, pred)

    pred = do_lin_reg_pred(X, y, X_fit)

    X_b = np.c_[np.ones((100, 1)), X]
    grad_desc_hyperparam = batch_grad_desc_best_theta(0.1, X_b, y, 100)
    print(f"BATCH GRAD DESCENT HYPERPARAMS: {grad_desc_hyperparam}")
    stoch_grad_desc_hyp = stochastic_grad_desc(X_b, y, 100)
    print(f"STOCHASTIC GRAD DESC. HYPERPARAMS: {stoch_grad_desc_hyp}")

    reg_with_stoch_desc(X, y)


def test_poly_reg():
    X, y, m = gen_quadratic_data()
    X_poly, poly_features = add_poly_features(X)

    X_new, X_new_poly = gen_new_X_data_for_pred(poly_features)

    y_pred = fit_lin_reg_on_poly_data(X_poly, y, X_new_poly)

    plot_poly_pred(X, y, X_new, y_pred)

def main():
    test_poly_reg()


if __name__=="__main__":
    main()