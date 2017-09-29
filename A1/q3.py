import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

BATCHES = 50


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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


# TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    m = X.shape[0]
    w_T = np.transpose(w).reshape(1, -1)
    gradient = 0.0
    for i in range(m):
        x = X[i].reshape(13, -1)
        gradient += 2 * x * (np.dot(w_T, x) - y[i])
    return gradient / m


def var(vec):
    mean = np.mean(vec)
    return np.sum((vec - mean) ** 2) / vec.shape[0]


def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    feature_count = w.shape[0]

    # Example usage
    K = 500
    batch_grad = 0.0
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad += lin_reg_gradient(X_b, y_b, w)
    batch_grad /= K
    total_grad = lin_reg_gradient(X, y, w)

    print("For m=50, K=500:")
    mse = np.mean((batch_grad - total_grad) ** 2)
    cosine = cosine_similarity(batch_grad[:, 0], total_grad[:, 0])
    print("Batch gradients:", batch_grad)
    print("Total gradients:", total_grad)
    print("Squared distance metric:", mse)
    print("Cosine similarity:", cosine)
    print('-------------------------------------------\n\n\n')


    M = 400
    vars = np.zeros(M * feature_count).reshape(M, feature_count)
    for m in range(M):
        grad = np.zeros(K * feature_count).reshape(K, feature_count)
        for i in range(K):
            X_b, y_b = batch_sampler.get_batch(m + 1)
            grad[i] = np.array(lin_reg_gradient(X_b, y_b, w)).reshape(1, -1)

        for f in range(feature_count):
            vars[m][f] = var(grad[:, f])

    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter(range(M), vars[:, i], s = 1)
        plt.xlabel('M')
        plt.ylabel('TARGET')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()