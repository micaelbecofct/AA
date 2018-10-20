#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.neighbors.kde import KernelDensity
from sklearn.base import BaseEstimator

class NaiveBayes_kde(BaseEstimator):
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        
    def separeteByclass(self,X, Y):
        unique_y = np.unique(Y)
        classes = {}
        logs = {}
        for i in unique_y:
            classes[i] = X[Y==i,:]
            logs[i] = np.log(classes[i].shape[0]/float(X.shape[0]))
        return classes, logs
        
    def fit(self, X, Y, condition=None):
        self.condition = condition
        classes, self.logs = self.separeteByclass(X,Y)
        self.kdes = {}
        for feature in range(X.shape[1]):
            self.kdes[feature]={}
            for c in classes:
                self.kdes[feature][c] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
                self.kdes[feature][c].fit(classes[c][:,[feature]])
       
    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred==y)/float(len(y))
    
    def predict(self, X):
        p = {}
        for c in self.logs:
            p[c] = np.ones(X.shape[0])* self.logs[c]
       
        for feature in range(X.shape[1]):
            for c in self.logs:
                p[c] = p[c] + self.kdes[feature][c].score_samples(X[:,[feature]])
        pred = np.zeros(X.shape[0])
        pred[self.condition(p)] = 1
        return pred
        

def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):
    best_h=1
    hs =[]
    errs=[]
    lowest=100000
    fit_params = {
    'condition': condition}
    for h in range(1,100,2):
        kde = NaiveBayes_kde(h/100)
        scores = cross_val_score(kde, X_r, Y_r, cv=Kf, fit_params=fit_params)
        va_err = 1-np.mean(scores)
        if va_err < lowest:
            lowest = va_err
            best_h = h
        errs.append(va_err)
        hs.append(h)

    fig = plt.figure(figsize=(8,8),frameon=False)
    plt.title('Naive Bayes') 
    plt.ylabel('Error')
    plt.xlabel('Bandwidths(x100)')
    plt.plot(hs, errs,'-',linewidth=3,label='Validation Error')
    plt.legend()
    fig.savefig('Naive_Bayes.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()

    kde = NaiveBayes_kde(best_h/100); kde.fit(X_r,Y_r, condition=condition)
    return 1-kde.score(X_t,Y_t), best_h, kde.predict(X_t)

def condition(p):
    return p[1]>p[0]










        
#implementar Logistic, Kneighborns
