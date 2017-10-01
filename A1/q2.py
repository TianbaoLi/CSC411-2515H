# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import math
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))
I = np.eye(d)

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    N_train = len(x_train)
    X_T = np.transpose(x_train)
    dist = l2(np.array(test_datum).T, x_train)
    dist = dist / (2 * tau ** 2)

    maxAi = - min(dist[0])
    dist_exp = np.zeros(dist.shape)
    for j in range(N_train):
        dist_exp[0][j] = np.exp(- dist[0][j] - maxAi)
    dist_exp_sum = np.sum(dist_exp[0])
    A = np.zeros(N_train * N_train).reshape(N_train, N_train)

    for j in range(N_train):
        A[j][j] = dist_exp[0][j] / dist_exp_sum

    w = np.linalg.solve(np.dot(np.dot(X_T, A), x_train) + I * lam, np.dot(np.dot(X_T, A), y_train))
    w = np.array(w).reshape(-1, 1)
    f_xw = np.dot(test_datum.T, w)

    return f_xw


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO: split into k-fold by idx
    global idx
    fold_size = int(math.ceil(1.0 * N / k))
    idx = [idx[i: min(i + fold_size, N)] for i in range(0, N, fold_size)]

    ## TODO: run run_on_fold() for k*taus_ammount times
    cross_validation = np.zeros(k * len(taus)).reshape(k, len(taus))
    losses = np.zeros((len(taus)))
    for i in range(k):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for n in range(N):
            if n in idx[i]:
                x_test.append(x[n])
                y_test.append(y[n])
            else:
                x_train.append(x[n])
                y_train.append(y[n])

        cross_validation[i] = run_on_fold(np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train), taus)
    for i in range(len(taus)):
        losses[i] = np.mean(cross_validation.T[i])
    return np.array(losses)


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200) #should be 1.0,3,200
    losses = run_k_fold(x,y,taus,k=5)
    print(losses)
    plt.plot(taus, losses)
    plt.xlabel('tau')
    plt.ylabel('loss value')
    plt.show()
    print("min loss = {}".format(losses.min()))
    print("arg loss = {}".format(losses.mean()))

