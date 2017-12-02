import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = - self.lr * grad + self.beta * self.vel
        params += self.vel
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count + 1)

    def addBias(selfself, X):
        n = X.shape[0]
        bias = np.ones((1, n))
        X = np.concatenate((bias.T, X), axis = 1)
        return X
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        X = self.addBias(X)
        n = X.shape[0]
        hinge_loss = np.zeros(n)
        for i in range(n):
            hinge_loss[i] = np.max(self.c * (1 - y[i] * np.vdot(self.w, X[i, :])), 0)
        return hinge_loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        X = self.addBias(X)
        n = X.shape[0]
        m = X.shape[1]
        gradients = np.zeros(m)
        for i in range(n):
            if y[i] * np.vdot(self.w, X[i, :]) < 1:
                gradients += - self.c * y[i] * X[i, :]
        gradients = 1.0 * gradients / n
        return gradients

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        X = self.addBias(X)
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n):
            pred[i] = 1 if np.vdot(self.w, X[i, :]) > 0 else -1
        return pred

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty, train_data.shape[1])
    w_history = np.zeros((iters + 1) * (train_data.shape[1] + 1)).reshape(iters + 1, train_data.shape[1] + 1)
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    for i in range(1, iters + 1):
        X_b, y_b = batch_sampler.get_batch()
        svm.w = optimizer.update_params(svm.w, svm.grad(X_b, y_b))
        w_history[i, :] = svm.w
    return svm

if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()

    print "Test SGD"
    gdo_0 = GDOptimizer(1.0, 0.0)
    gdo_9 = GDOptimizer(1.0, 0.9)
    opt_history = np.zeros(402).reshape(2, 201)
    opt_history[0] = optimize_test_function(gdo_0, 10.0, 200)
    opt_history[1] = optimize_test_function(gdo_9, 10.0, 200)
    plt.plot(range(200 + 1), opt_history[0], label = 'beta = 0.0')
    plt.plot(range(200 + 1), opt_history[1], label = 'beta = 0.9')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('W')
    plt.show()

    print "Test SVM"
    m = 100
    T = 500
    C = 1.0

    gdo_0 = GDOptimizer(0.05, 0.0)
    svm_0 = optimize_svm(train_data, train_targets, C, gdo_0, m, T)
    print "SVM 0 training loss = ", svm_0.hinge_loss(train_data, train_targets)
    print "SVM 0 test loss = ", svm_0.hinge_loss(test_data, test_targets)
    train_pred = svm_0.classify(train_data)
    print "SVM 0 training accuracy = ", (train_pred == train_targets).mean()
    test_pred = svm_0.classify(test_data)
    print "SVM 0 test accuracy = ", (test_pred == test_targets).mean()
    plt.imshow(svm_0.w[: -1].reshape(28, 28), cmap = 'gray')
    plt.show()

    gdo_1 = GDOptimizer(0.05, 0.1)
    svm_1 = optimize_svm(train_data, train_targets, C, gdo_1, m, T)
    print "SVM 1 training loss = ", svm_1.hinge_loss(train_data, train_targets)
    print "SVM 1 test loss = ", svm_1.hinge_loss(test_data, test_targets)
    train_pred = svm_1.classify(train_data)
    print "SVM 1 training accuracy = ", (train_pred == train_targets).mean()
    test_pred = svm_1.classify(test_data)
    print "SVM 1 test accuracy = ", (test_pred == test_targets).mean()
    plt.imshow(svm_1.w[: -1].reshape(28, 28), cmap = 'gray')
    plt.show()