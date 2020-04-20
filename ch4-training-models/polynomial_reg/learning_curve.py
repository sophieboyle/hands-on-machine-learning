from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
    @brief Plot the learning curve of a given model.
    @param model to fit
    @param X array of feature values to fit upon
    @param y array of labels
"""
def plot_learning_curves(model, X, y):
    # Divide up training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                    random_state=10)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        # Incrementally increase the data set which the model
        # fits on by one each time
        model.fit(X_train[:m], y_train[:m])
        # Make a prediction using the current training slice
        y_train_predict = model.predict(X_train[:m])
        # Make a prediction on the validation set
        y_val_predict = model.predict(X_val)
        # Append the MSE for this iteration
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    # Plot the learning curve
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.axis([0, 80, 0, 3])
    plt.show()