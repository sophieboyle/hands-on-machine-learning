from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


def get_img(filename):
    """Returns image of the given filename in
    the form of a 3D array.

    NOTE: The image is in the shape:
    (height, width, no. of colour channels)

    Arguments:
        filename {string} -- Name of the file.

    Returns:
        array -- Image in the form of a 3D array.
    """
    image = imread(os.path.join("images", filename))
    return image


if __name__ == "__main__":
    image = get_img("ladybug.png")
    # Gets arrays of RGB colour values
    X = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=8).fit(X)
    # Segment the pixels into their mean colours
    seg_img = kmeans.cluster_centers_[kmeans.labels_]
    # Regain the shape of the original image
    seg_img = seg_img.reshape(image.shape)

    # NOTE: It basically looks like the posterise filter
    # Notice that when decreasing the number of clusters,
    # less colours can be identified (such as that of the
    # red ladybug)
    plt.imshow(seg_img)
    plt.show()