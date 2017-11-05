'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        dist = self.l2_distance(test_point)

        noTies = False
        while noTies == False:
            mink = np.argpartition(dist, k)[:k]
            counts = np.bincount(self.train_labels[mink].astype(int))
            digit = np.argwhere(counts == np.amax(counts))
            if len(digit) > 1:
                k = k - 1
            else:
                noTies = True
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    for i in range(15):
        print i+1, classification_accuracy(knn, i+1, test_data, test_labels)
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    accuracy = 0.0
    for i in range(len(eval_data)):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == int(eval_labels[i]):
            accuracy = accuracy + 1
    return accuracy / len(eval_data)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # K = 1:
    print "K = 1:", classification_accuracy(knn, 1, test_data, test_labels)
    # K = 15:
    print "K = 15:", classification_accuracy(knn, 15, test_data, test_labels)


if __name__ == '__main__':
    main()