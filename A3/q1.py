'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the baseline model
    tfidf_train = (tfidf_train>0).astype(int)
    tfidf_test = (tfidf_test>0).astype(int)

    model = BernoulliNB()
    model.fit(tfidf_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(tfidf_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def logistic(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the logistic regression model
    tfidf_train = (tfidf_train > 0).astype(int)
    tfidf_test = (tfidf_test > 0).astype(int)
    tfidf_train, tfidf_validation, train_labels, validation_labels = train_test_split(tfidf_train, train_labels, test_size = 0.2)

    model = LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-4)
    Cs = [1, 5, 10, 20, 30, 40, 50]
    train_accuracy = []
    valid_accuracy = []
    for c in Cs:
        model.set_params(C = c)
        model.fit(tfidf_train, train_labels)
        train_pred = model.predict(tfidf_train)
        train_accuracy.append((train_pred == train_labels).mean())
        validation_pred = model.predict(tfidf_validation)
        valid_accuracy.append((validation_pred == validation_labels).mean())

    opt_C_index = int(np.argmax(valid_accuracy))
    print('Optimal C for logistic regression = {}'.format(Cs[opt_C_index]))
    print('Logistic regression train accuracy = {}'.format(train_accuracy[opt_C_index]))
    print('Logistic regression validation accuracy = {}'.format(valid_accuracy[opt_C_index]))
    test_pred = model.predict(tfidf_test)
    print('Logistic regression test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def SGD(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the stochastic gradient descent model
    model = SGDClassifier(alpha = 0.01, tol = 1e-4)
    model.fit(tfidf_train, train_labels)

    train_pred = model.predict(tfidf_train)
    print('SGD train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('SGD test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def SVM(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the support vector machine model
    tfidf_train = (tfidf_train > 0).astype(int)
    tfidf_test = (tfidf_test > 0).astype(int)
    tfidf_train, tfidf_validation, train_labels, validation_labels = train_test_split(tfidf_train, train_labels, test_size = 0.2)

    model = svm.SVC(decision_function_shape = 'ovo', kernel = 'linear', tol = 1e-4)
    Cs = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
    train_accuracy = []
    valid_accuracy = []
    for c in Cs:
        model.set_params(C = c)
        model.fit(tfidf_train, train_labels)
        train_pred = model.predict(tfidf_train)
        train_accuracy.append((train_pred == train_labels).mean())
        validation_pred = model.predict(tfidf_validation)
        valid_accuracy.append((validation_pred == validation_labels).mean())

    opt_C_index = int(np.argmax(valid_accuracy))
    print('Optimal C for logistic regression = {}'.format(Cs[opt_C_index]))
    print('SVM train accuracy = {}'.format(train_accuracy[opt_C_index]))
    print('SVM validation accuracy = {}'.format(valid_accuracy[opt_C_index]))
    test_pred = model.predict(tfidf_test)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def KNN(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the K nearest neighbors model
    tfidf_train = (tfidf_train > 0).astype(int)
    tfidf_test = (tfidf_test > 0).astype(int)
    tfidf_train, tfidf_validation, train_labels, validation_labels = train_test_split(tfidf_train, train_labels, test_size = 0.2)

    model = neighbors.KNeighborsClassifier(n_neighbors = 1)
    Ks = range(1, 101, 10)
    train_accuracy = []
    valid_accuracy = []
    for k in Ks:
        model.set_params(n_neighbors = k)
        model.fit(tfidf_train, train_labels)
        train_pred = model.predict(tfidf_train)
        train_accuracy.append((train_pred == train_labels).mean())
        validation_pred = model.predict(tfidf_validation)
        valid_accuracy.append((validation_pred == validation_labels).mean())

    opt_K_index = int(np.argmax(valid_accuracy))
    print('Optimal K for K nearest neighbors = {}'.format(Ks[opt_K_index]))
    print('KNN train accuracy = {}'.format(train_accuracy[opt_K_index]))
    print('KNN validation accuracy = {}'.format(valid_accuracy[opt_K_index]))
    test_pred = model.predict(tfidf_test)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tfidf, test_tfidf, feature_names = tf_idf_features(train_data, test_data)

    print('### BernoulliNB baseline ###')
    bnb_model = bnb_baseline(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Logistic regression ###')
    #logistic_model = logistic(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Stochastic gradient descent ###')
    #SGD_model = SGD(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Support vector machine ###')
    #SVM_model = SVM(train_tfidf, train_data.target, test_tfidf, test_data.target)
    print('### K nearest neighbors ###')
    KNN_model = KNN(train_tfidf, train_data.target, test_tfidf, test_data.target)