from data_handling import get_data, load_data, display_data, split_data, split_data_by_id, do_stratified_sampling, play_with_data
from transformers import CombinedAttributesAdder
from helper import display_scores
from scipy.stats import expon, reciprocal

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
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
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

    print("Finished preparing data")

    svr_reg = SVR()

    # # Try a support vector machine regressor
    # param_grid = [
    #         {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
    #         {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
    #          'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    #     ]
    # 
    # grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
    #                             scoring="neg_mean_squared_error",
    #                             return_train_score=True)
    # grid_search.fit(housing_prepared, housing_labels)
    # 
    # # Best svr score
    # best_svr_score = np.sqrt(-grid_search.best_score_)
    # print(f"Best SVR Estimator Score: {best_svr_score}") 

    # Using a randomized search instead of a grid search
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

    rnd_search = RandomizedSearchCV(svr_reg, param_distribs,
                                    n_iter=50, cv=5, scoring="neg_mean_squared_error",
                                    verbose=2, random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    best_svr_score = np.sqrt(-rnd_search.best_score_)
    print(f"Best SVR Estimator Score: {best_svr_score}")
    # NOTE: This results in a better score than that of the grid search

if __name__=="__main__":
    main()