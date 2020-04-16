from data_handling import get_data, load_data, display_data, split_data, split_data_by_id, do_stratified_sampling, play_with_data
from transformers import CombinedAttributesAdder
from helper import display_scores

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy import stats

def main():
    # Get and load data
    get_data()
    housing = load_data()
    # display_data(housing)

    # Perform and split by strata
    strat_train_set, strat_test_set = do_stratified_sampling(housing)
    
    # Using the training set, play with the data
    # play_with_data(strat_train_set.copy())

    # Split data into predictors and labels
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Use an imputer to fill in missing values
    # We will fill in these values with the median
    imputer = SimpleImputer(strategy="median")
    # Get dataframe of only numerical vals
    housing_num = housing.drop("ocean_proximity", axis=1)

    # Let the imputer estimate based on the numerical housing vals
    imputer.fit(housing_num)
    # NOTE: The median of each attribute is stored in imputer.statistics_
    # Use trained imputer to fill in gaps by transforming the data
    X = imputer.transform(housing_num)
    # Insert np array into pandas DataFrame
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                                index=housing_num.index)

    # Convert categorical attribute to numerical attribute
    housing_cat = housing[["ocean_proximity"]]
    # Use one-hot encoding instead of ordinal encoding
    # as the categories are not ordered.
    cat_encoder = OneHotEncoder()

    # NOTE: This gives a scipy array which stores the location
    # of the "hot" encoding (instead of potentially storing
    # many many "cold" encodings (0's))
    # NOTE: Categories are stored in ordinal_encoder.categories_
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    # Adding combinational attributes
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    # Pipeline for transformations on numerical values
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # It is also possible to perform all of the above transformations
    # in one go
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # This is the final set of training data
    housing_prepared = full_pipeline.fit_transform(housing)

    # Fit the linear regression model on prepared data
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Do some testing
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))

    # Get metrics
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    # Due to the above results being unsatisfactory
    # Try a decision tree regressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # Now do some testing on the tree regression model
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

    # The above testing gives no error
    # Cross validation is performed on 10 folds (training and validating
    # 10 times, choosing a different fold for validation each time
    # and training on the remaining fold)
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
    # As cross validation expect to use a utility function instead of a
    # cost function (whereas we want to use a cost function), we must
    # flip the sign of the scores.
    tree_rmse_scores = np.sqrt(-scores)

    # Double check against cross validation on the linear reg. model
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)

    print("TREE RSME SCORES")
    display_scores(tree_rmse_scores)

    print("LINEAR REG RMSE SCORES")
    display_scores(lin_rmse_scores)

    # This shows that the Decision Tree is overfitting
    # Therefore we try the Random Forest Regressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)

    print("RANDOM FOREST REG RMSE SCORES")
    display_scores(forest_rmse_scores)

    # Fine-tuning by automatically searching for hyperparams
    # Grid indicates to try firstly all permutations of the first dict
    # followed by the permutations of options in the second dict.
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features":[2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    # We use five-fold cross validation
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                scoring="neg_mean_squared_error",
                                return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    # The best parameters are found using:
    print(f"Best hyperparams: {grid_search.best_params_}")
    # The best estimator:
    print(f"Best Estimator: {grid_search.best_estimator_}")
    # The evaluation scores:
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # Examine the relative importance of each attribute for accurate predictions
    feature_importances = grid_search.best_estimator_.feature_importances_
    # Displaying the importance scores next to their attribute names
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))
    # NOTE: The above may indicate which features may be dropped

    # Evaluation on test set
    # Select the best estimator found by the grid search as the final model
    final_model = grid_search.best_estimator_

    # Separate test set into predictors and labels
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # NOTE: Only transform test data, DO NOT FIT the model on test data
    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    # Compute 95% confidence interval
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    np.sqrt(stats.t.interval(confidence, len(squared_errors) -1,
                            loc=squared_errors.mean(),
                            scale=stats.sem(squared_errors)))

    # Try a support vector machine regressor
    param_grid = [
            {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
            {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
             'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ]
    
    svr_reg = SVR()
    grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
                                scoring="neg_mean_squared_error",
                                return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    
    # Best svr score
    best_svr_score = np.sqrt(-grid_search.best_score_)
    print(f"Best SVR Estimator Score: {best_svr_score}") 


if __name__=="__main__":
    main()