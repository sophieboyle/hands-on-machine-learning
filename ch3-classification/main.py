from data import get, view_example
import os

def main():
    mnist = get()

    # X.shape = (70000, 784), 70,000 isntances with 784 features
    # Each feature represents a pixel (28x28 img)
    # y.shape = (70000), contains labels for X
    X, y = mnist["data"], mnist["target"]
    view_example(X)

if __name__=="__main__":
    main()