from sklearn.base import BaseEstimator, TransformerMixin
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
    Transformer class to only retain a given
    list of important attributes in a dataframe.
"""
class SelectImportantAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, important_attribs):
        self.important_attribs = important_attribs
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        pass