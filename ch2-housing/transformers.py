from sklearn.base import BaseEstimator, TransformerMixin
from helper import top_importances
import numpy as np

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

"""
    Transformer class to add combined attributes:
    rooms per household, population per household,
    and (optional) bedrooms per room.
"""
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        # Example of hyperparameter
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, households_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

"""
    Transformer class to only retain a given specific
    number of most important features in data.
"""
class SelectImportantFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    
    def fit(self, X):
        # Get the indices of the top features
        self.top_indices_ = top_importances(self.feature_importances, self.k)
        return self
    
    def transform(self, X):
        # Return the data with only the columns
        # indicative of the top features
        return X[:, self.top_indices_]