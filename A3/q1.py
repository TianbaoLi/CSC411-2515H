'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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

def mnb(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the multinomial naive bayes model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    As = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2, 5, 10]
    scores = []
    for a in As:
        model = MultinomialNB(alpha = a)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_A_index = int(np.argmax(scores))
    opt_A = As[opt_A_index]
    print('Optimal Alpha for multinomial naive bayes = {}'.format(opt_A))
    model = MultinomialNB(alpha = opt_A)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('MultinomialNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('MultinomialNB test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def logistic(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the logistic regression model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    Cs = [0.01, 0.1, 1, 50, 100, 500, 1000, 2000]
    scores = []
    for c in Cs:
        model = LogisticRegression(C = c)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_C_index = int(np.argmax(scores))
    opt_C = Cs[opt_C_index]

    print('Optimal C for logistic regression = {}'.format(opt_C))
    model = LogisticRegression(C = opt_C)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('Logistic regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('Logistic regression test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def SGD(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the stochastic gradient descent model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    As = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2]
    scores = []
    for a in As:
        model = SGDClassifier(alpha = a, max_iter = 100, tol = 1e-3, learning_rate = 'optimal')
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_A_index = int(np.argmax(scores))
    opt_A = As[opt_A_index]
    print('Optimal Alpha for stochastic gradient descent = {}'.format(opt_A))
    model = MultinomialNB(alpha=opt_A)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('SGD train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('SGD test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def SVM(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the support vector machine model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    Cs = [0.01, 0.1, 1, 5]
    scores = []
    for c in Cs:
        model = svm.LinearSVC(penalty = "l1", dual = False, tol = 1e-3, C = c)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_C_index = int(np.argmax(scores))
    opt_C = Cs[opt_C_index]

    print('Optimal C for support vector machine = {}'.format(opt_C))
    model = LogisticRegression(C=opt_C)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def KNN(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the K nearest neighbors model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    Ks = range(1, 11)
    scores = []
    for k in Ks:
        model = neighbors.KNeighborsClassifier(n_neighbors = k)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_K_index = int(np.argmax(scores))
    opt_K = Ks[opt_K_index]

    print('Optimal K for K nearest neighbors = {}'.format(opt_K))
    model = neighbors.KNeighborsClassifier(n_neighbors = opt_K)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('KNN train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('KNN test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def decision_tree(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the decision tree model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    Ds = range(1, 1201, 300)
    scores = []
    for d in Ds:
        model = DecisionTreeClassifier(max_depth = d)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_D_index = int(np.argmax(scores))
    opt_D = Ds[opt_D_index]

    print('Optimal K for decision tree = {}'.format(opt_D))
    model = DecisionTreeClassifier(max_depth = opt_D)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('Decision tree train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('Decision tree test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def NN(tfidf_train, train_labels, tfidf_test, test_labels):
    # training the neural network model
    splits = 5
    kf = KFold(splits, shuffle = True, random_state = 0)
    As = [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1]
    scores = []
    for a in As:
        model = MLPClassifier(alpha = a, early_stopping = True)
        score = cross_val_score(model, tfidf_train, train_labels, cv = kf)
        scores.append(np.mean(score))

    opt_A_index = int(np.argmax(scores))
    opt_A = As[opt_A_index]
    print('Optimal Alpha for neural network = {}'.format(opt_A))
    model = MLPClassifier(alpha = opt_A, early_stopping = True)
    model.fit(tfidf_train, train_labels)
    train_pred = model.predict(tfidf_train)
    print('Neural network train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tfidf_test)
    print('Neural network test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tfidf, test_tfidf, feature_names = tf_idf_features(train_data, test_data)

    print('### BernoulliNB baseline ###')
    bnb_model = bnb_baseline(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### MultinomialNB baseline ###')
    #mnb_model = mnb(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Logistic regression ###')
    #logistic_model = logistic(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Stochastic gradient descent ###')
    #SGD_model = SGD(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Support vector machine ###')
    #SVM_model = SVM(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### K nearest neighbors ###')
    #KNN_model = KNN(train_tfidf, train_data.target, test_tfidf, test_data.target)
    #print('### Decision tree ###')
    #decision_tree_model = decision_tree(train_tfidf, train_data.target, test_tfidf, test_data.target)
    print('### Neural network ###')
    NN_model = NN(train_tfidf, train_data.target, test_tfidf, test_data.target)