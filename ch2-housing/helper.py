import numpy as np

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

"""
    @brief Given a dataframe and the maximum number
           of features we want to retain in our model,
           return the indices of the top features.
    @param nparray of top feature data
    @param k number of features we wish to identify
    @return sorted np array of indices
"""
def top_importances(data, k):
    # Partition array into greatest k instances
    # and then return the slice containing these.
    return np.sort(np.argpartition(np.array(data), -k)[-k:])