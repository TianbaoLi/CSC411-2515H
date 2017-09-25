from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def summarize_data(X, y, features):
    print("Number of data points: {}".format(len(X)))
    print("Dimensions: {}".format(len(features)))
    print("Features: {}".format(features))
    print("Mean house price: {}".format(np.mean(y)))
    print("Standard deviation of house price: {}".format(np.std(y)))


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter([x[i] for x in X], y, s = 1)
        plt.xlabel(features[i])
        plt.ylabel('TARGET')

    plt.tight_layout()
    plt.show()

def split_data(X, y, training_ratio = 0.2):
    test_chosen = np.random.choice(len(X), int(len(X) * training_ratio))
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
    mae = np.mean(abs(y - f_xw))
    r2 = 1 - np.sum((y - f_xw) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return mse, mae, r2


def sklearn_method(training_set_x, test_set_x, training_set_y, test_set_y):
    lm = LinearRegression()
    lm.fit(training_set_x, training_set_y)
    Y_pred = lm.predict(test_set_x)
    mse = sklearn.metrics.mean_squared_error(test_set_y, Y_pred)
    mae = sklearn.metrics.mean_absolute_error(test_set_y, Y_pred)
    r2 = sklearn.metrics.r2_score(test_set_y, Y_pred)
    return mse, mae, r2

def main():
    # Load the data
    X, y, features = load_data()

    # Summarize data
    summarize_data(X, y, features)

    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    training_set_x, test_set_x, training_set_y, test_set_y = split_data(X, y)

    # Fit regression model
    w = fit_regression(training_set_x, training_set_y)
    feature_weight = []
    for i in range(len(features)):
        feature_weight.append((features[i], w[i]))
    print("Features vs weight: {}".format(feature_weight))

    # Compute fitted values, MSE, etc.
    mae, mse, r2 = test(test_set_x, test_set_y, w)
    print("MSE MAE R2: {}".format((mae, mse, r2)))

    sklearn_mae, sklearn_mse, r2_sklearn = sklearn_method(training_set_x, test_set_x, training_set_y, test_set_y)
    print("MSE MAE R2 by sklearn linear regression: {}".format((sklearn_mae, sklearn_mse, r2_sklearn)))

if __name__ == "__main__":
    main()

