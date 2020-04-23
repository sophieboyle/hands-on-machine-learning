from lin_reg.closed_form_sol import *
from lin_reg.lin_reg import *
from lin_reg.grad_descent import *
from polynomial_reg.polynomial_reg import *
from polynomial_reg.learning_curve import *
from regularisation.ridge_reg import *
from regularisation.lasso_reg import *
from regularisation.elastic_net import *
from regularisation.early_stopping import *
from logistic_reg.iris import *
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


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

    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)

    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression())
    ])

    plot_learning_curves(polynomial_regression, X, y)


def test_regularisation():
    X, y = gen_reg_ex_data()

    ridge_reg = do_ridge_reg(X, y)
    print(ridge_reg.predict([[1.5]]))

    sgd_reg = do_ridge_reg_using_sgd(X, y)
    print(sgd_reg.predict([[1.5]]))

    lasso_reg = do_lasso_reg(X, y)
    print(lasso_reg.predict([[1.5]]))

    sgd_lasso_reg = do_lasso_reg_with_sgd(X, y)
    print(sgd_lasso_reg.predict([[1.5]]))

    elastic_net = do_elastic_net(X, y, 0.5)
    print(elastic_net.predict([[1.5]]))

    X_train, X_val, y_train, y_val = gen_and_split_data(100)
    X_train_poly_scaled, X_val_poly_scaled = prep_data(X_train, X_val)
    sgdr_es = early_stopping_sgdr(X_train, X_val, y_train, y_val)
    
    

def main():
    iris_main()


if __name__=="__main__":
    main()