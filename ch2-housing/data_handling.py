import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

dl = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
house_path = os.path.join("datasets", "housing")
house_url = dl + "datasets/housing/housing.tgz"


"""
    @brief Downloads dataset and unzips it
    @param url to download from
    @param path to download to
"""
def get_data(url=house_url, path=house_path):
    # Overwrite data if it already exists
    os.makedirs(path, exist_ok=True)
    tgz = os.path.join(path, "housing.tgz")
    urllib.request.urlretrieve(url, tgz)
    # Get downloaded zip and unzip it
    target = tarfile.open(tgz)
    target.extractall(path=path)
    target.close()


"""
    @brief Load csv as panda dataFrame
    @param path to read the csv file from
    @return pandas.DataFrame of data contained by file
"""
def load_data(path=house_path):
    csv = os.path.join(path, "housing.csv")
    return pd.read_csv(csv)


"""
    @brief Display given data
    @param DataFrame to be displayed
"""
def display_data(housing):
    # Viewing data
    print(housing.head())
    # NOTE: Some instances are missing total_bedrooms value
    print(housing.info())
    # NOTE: Ocean is a categorical object
    print(housing["ocean_proximity"].value_counts())
    # Describe numerical values
    print(housing.describe())
    # Present histogram
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()


"""
    @brief Splits the data into a training and test set.
    @param pandas.DataFrame of data to split.
    @param floating point test_ratio of the number of test
           instances in comparison to training instances.
    @return pandas.DataFrame of training set and testing set
            respectively.
"""
def split_data(data, test_ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    # Slice the data into different sets of indices
    test_set = shuffled[:test_set_size]
    train_set = shuffled[test_set_size:]
    return data.iloc[train_set], data.iloc[test_set]


"""
    @brief Compute hash of test set instances
    @param id of the instance
    @param floating point ratio of the test set
"""
def test_set_check(id, test_ratio):
    return crc32(np.int64(id)) & 0xffffffff < test_ratio * 2**32


"""
    @brief Splits data using id hashing.
    @param pandas.DataFrame of the data to split.
    @param floating point ratio of training data to test data.
    @param string the name of the column to use as an ID.
    @return pandas.DataFrame of training set and testing set
            respectively.
"""
def split_data_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


"""
    @brief Perform stratified sampling and split on data
    @param housing data to be sampled and split
    @return DataFrames for the training and testing sets respectively
"""
def do_stratified_sampling(housing):
    # To perform stratified sampling, divide up median income
    # into strata
    housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    
    # Display histogram of categories
    # housing["income_cat"].hist()
    # plt.show()

    # Perform stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Now remove income_cat feature to restore data's state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    return strat_train_set, strat_test_set


"""
    Playing with a copy of the training data
"""
def play_with_data(housing):
    # Low alpha used to view density of point distribution 
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    # Looking particularly at the housing prices
    # Radius (s) representative of population
    # Colour representing price (c: blue->low, red->high)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=housing["population"]/100, label="population", figsize=(10,7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()

    # Standard correlation coefficient
    corr_matrix = housing.corr()
    # View how each attribute correlates with median house value
    # NOTE: Only measures linear correlations
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # We get the most interestingly correlated attributes
    # where their correlation is not close to 0
    attributes = ["median_house_value", "median_income",
                    "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    plt.show()

    # From the above, we can see that median income seems to have
    # an interesting relationship with median house value
    # NOTE: The plot shows some straight lines of districts that
    #       should probably be removed.
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                alpha=0.1)
    plt.show()

    # Given insights by viewing the data
    # we could add some more interesting combinations
    # of attributes to our data
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


