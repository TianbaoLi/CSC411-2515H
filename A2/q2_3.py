'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    K = eta.shape[0]
    d = eta.shape[1]

    for k in range(K):
        k_index = np.argwhere(train_labels == k)
        k_digits = train_data[k_index].reshape(-1, d)
        for j in range(d):
            eta[k][j] = (np.sum(k_digits[:, j]) + 1.0) / (k_digits.shape[0] + 2.0)

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    parameters = np.zeros((10, 8, 8))
    for i in range(10):
        img_i = class_images[i]
        # ...
        parameters[i] = img_i.reshape(8, 8)
    all_concat = np.concatenate(parameters, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    K = eta.shape[0]
    d = eta.shape[1]
    generated_data = np.zeros((10, 64))

    for k in range(K):
        for j in range(d):
            generated_data[k][j] = np.random.binomial(1, eta[k][j])

    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    N = bin_digits.shape[0]
    d = bin_digits.shape[1]
    gen_likelihood = np.zeros((N, 10))

    for n in range(N):
        for k in range(10):
            gen_likelihood[n][k] = 1.0
            for j in range(64):
                gen_likelihood[n][k] = gen_likelihood[n][k] * np.power(eta[k][j], bin_digits[n][j]) * np.power(1 - eta[k][j], 1 - bin_digits[n][j])
            gen_likelihood[n][k] = np.log(gen_likelihood[n][k])
    return gen_likelihood

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    N = bin_digits.shape[0]
    con_likelihood = np.zeros((N, 10))
    likelihood = generative_likelihood(bin_digits, eta)

    for n in range(N):
        total_probability = np.log(np.sum(np.exp(likelihood[n][:])))
        for k in range(10):
            con_likelihood[n][k] = likelihood[n][k] + np.log(1.0 / 10) - total_probability

    return con_likelihood

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    avg = np.zeros(10)
    true_label_count = np.zeros(10)
    N = bin_digits.shape[0]
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    for n in range(N):
        true_label = int(labels[n])
        true_label_count[true_label] = true_label_count[true_label] + 1
        avg[true_label] = avg[true_label] + cond_likelihood[n][true_label]

    for i in range(10):
        avg[i] = avg[i] / true_label_count[i]

    return avg

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    N = bin_digits.shape[0]
    prediction = np.zeros((N))
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    for n in range(N):
        prediction[n] = np.argmax(cond_likelihood[n][:])

    return prediction

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)

    train_avg = avg_conditional_likelihood(train_data, train_labels, eta)
    print "Training data avg conditional likelihood:", train_avg

    test_avg = avg_conditional_likelihood(test_data, test_labels, eta)
    print "Test data avg conditional likelihood:", test_avg

if __name__ == '__main__':
    main()
