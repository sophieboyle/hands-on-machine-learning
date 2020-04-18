from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


"""
    @brief Retrieve the specified data
           Gets MNIST by default.
    @param string title of the dataset
    @return dictionary of data with keys:
            ['data', 'target', 'frame', 'feature_names', 
            'target_names', 'DESCR', 'details', 'categories', 'url']
"""
def get(data_title='mnist_784'):
    data = fetch_openml(data_title, version=1)
    return data

"""
def save(data, title):
    with open('datasets/' + title + '.p', 'w') as f:
        pickle.dump(data, f)


def load(title):
    with open('datasets/' + title + '.p', 'w') as f:
        data = pickle.load(f)
    return f
"""

"""
    @brief Given the data array X, display an
           example of the first digit of the MNIST
           dataset in image form.
    @param Data array X
"""
def view_example(X):
    # Get image data
    some_digit = X[0]
    # Reshape to a 28 x 28 array
    some_digit_img = some_digit.reshape(28, 28)
    # Display
    plt.imshow(some_digit_img, cmap="binary")
    plt.axis("off")
    plt.show()