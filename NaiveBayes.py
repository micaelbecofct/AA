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
        
    def fit(self, X, Y, sample_weight=None, corrector=None):
        self.corrector = corrector
        classes, self.logs = self.separeteByclass(X,Y)
        self.kdes = {}
        for feature in range(X.shape[1]):
            self.kdes[feature]={}
            for c in classes:
                self.kdes[feature][c] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
                self.kdes[feature][c].fit(classes[c][:,[feature]])
       
    def score(self, X, y):
        p = {}
        for c in self.logs:
            p[c] = np.ones(X.shape[0])* self.logs[c]
       
        for feature in range(X.shape[1]):
            for c in self.logs:
                p[c] = p[c] + self.kdes[feature][c].score_samples(X[:,[feature]])
        pred = np.zeros(X.shape[0])
        pred[self.corrector(p)] = 1
        return np.sum(pred==y)/float(len(y))
    
    def predict(self, X):
        p = {}
        for c in self.logs:
            p[c] = np.ones(X.shape[0])* self.logs[c]
       
        for feature in range(X.shape[1]):
            for c in self.logs:
                p[c] = p[c] + self.kdes[feature][c].score_samples(X[:,[feature]])
        pred = np.zeros(X.shape[0])
        pred[self.corrector(p)] = 1
        return pred
        

def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):
    best_bandwidth=1
    bandwidths =[]
    errs=[]
    lowest=100000
    fit_params = {
    'corrector': corrector}
    for bw in range(1,100,2):
        kde = NaiveBayes_kde(bw/100)
        scores = cross_val_score(kde, X_r, Y_r, cv=Kf, fit_params=fit_params)
        va_err = 1-np.mean(scores)
        if va_err < lowest:
            lowest = va_err
            best_bandwidth = bw
        errs.append(va_err)
        bandwidths.append(bw)
    plt.figure(figsize=(8,8),frameon=False)
    plt.plot(bandwidths, errs,'-',linewidth=3)
    plt.show()
    plt.close()

    kde = NaiveBayes_kde(best_bandwidth/100); kde.fit(X_r,Y_r, corrector=corrector)
    return 1-kde.score(X_t,Y_t), best_bandwidth, kde.predict(X_t)

def corrector(p):
    return p[1]>p[0]










        
#implementar Logistic, Kneighborns
