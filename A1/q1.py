from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter([x[i] for x in X], y, s = 1)
        plt.xlabel(features[i])
        plt.ylabel('y')

    plt.tight_layout()
    plt.show()

def split_data(X, y):
    test_chosen = np.random.choice(len(X), int(len(X) * 0.2))
    training_set_x = []
    test_set_x = []
    training_set_y = []
    test_set_y = []
    for i in range(len(X)):
        if i in test_chosen:
            test_set_x.append(X[i])
        else:
            training_set_x.append((X[i]))
    for i in range(len(y)):
        if i in test_chosen:
            test_set_y.append(y[i])
        else:
            training_set_y.append((y[i]))
    return training_set_x, test_set_x, training_set_y, test_set_y


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    bias = np.ones((1, len(X)))
    X = np.concatenate((bias.T, X), axis = 1)
    X_T = np.transpose(X)
    W = np.linalg.solve(np.dot(X_T, X), np.dot(X_T, Y))
    return W


def test(X, y, w):
    bias = np.ones((1, len(X)))
    X = np.concatenate((bias.T, X), axis=1)
    f_xw = np.dot(X, np.transpose(w))
    mse = np.mean((y - f_xw) ** 2)
    return mse


def main():
    # Load the data
    X, y, features = load_data()
    #print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    training_set_x, test_set_x, training_set_y, test_set_y = split_data(X, y)

    # Fit regression model
    w = fit_regression(X, y)
    print w

    # Compute fitted values, MSE, etc.
    MSE = test(X, y, w)
    print MSE

if __name__ == "__main__":
    main()

