from closed_form_sol import *
from lin_reg import *
from grad_descent import *

def main():
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

if __name__=="__main__":
    main()