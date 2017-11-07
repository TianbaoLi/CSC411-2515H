'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0, 10):
        # Compute mean of class i
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        for j in range(i_digits.shape[1]):
            means[i][j] = np.sum(i_digits[:, j]) / i_digits.shape[0]
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        i_digits = i_digits.T
        Ei = means[i]
        for j in range(64):
            for k in range(64):
                covariances[i][j][k] = np.dot((i_digits[j] - Ei[j]).T, (i_digits[k] - Ei[k])) / (i_digits.shape[1] - 1)
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_diagonals = np.zeros((10, 8, 8))
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        log_diagonals[i] = np.log(cov_diag).reshape(8, 8)

    all_concat = np.concatenate(log_diagonals, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    N = digits.shape[0]
    d = digits.shape[1]
    gen_likelihood = np.zeros(N, 10)

    for n in range(N):
        for i in range(10):
            gen_likelihood[n][i] = -(1.0 * d / 2 * np.log(2 * np.pi)) * (1 / 2 * np.log(np.linalg.det(covariances[i]))) * (-1.0 / 2 * np.cov(np.cov((digits[n] - means[i]).T, np.linalg.inv(covariances[i]))), (digits[n] - means[i]))
    return gen_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    N = digits.shape[0]
    con_likelihood = np.zeros(N, 10)
    likelihood = generative_likelihood(digits, means, covariances)

    for n in range(N):
        total_probability = 0.0
        for i in range(10):
            total_probability = total_probability + likelihood[n][i]
        for i in range(10):
            con_likelihood[n][i] = likelihood[n][i] * (1.0 / 10) / total_probability

    return con_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    avg = np.zeros(10)
    true_label_count = np.zeros(10)
    N = digits.shape[0]
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for n in range(N):
        true_label = labels[n]
        true_label_count[true_label] = true_label_count[true_label] + 1
        avg[true_label] = avg[true_label] + cond_likelihood[n][true_label]

    for i in range(10):
        avg[i] = avg[i] / true_label_count[i]

    return cond_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)

    # Evaluation
    print avg_conditional_likelihood(train_data, train_labels, means, covariances)

if __name__ == '__main__':
    main()