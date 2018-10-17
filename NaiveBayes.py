#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:09:21 2018

@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
import matplotlib.pyplot as plt

from math import log
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors.kde import KernelDensity


def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5)
    best_bandwidth=1
    bandwidths =[]
    errs=[]
    lowest=100000
    for bandwidth in range(1,100,2):
        scores = gscores(kf, X_r, Y_r,bandwidth/100)
        va_err = 1-np.mean(scores)
        print(va_err)
        if va_err<lowest:
           lowest = va_err
           best_bandwidth= bandwidth
        errs.append(va_err)
        bandwidths.append(bandwidth/100)
    errs = np.array(errs)
    plt.figure(figsize=(8,8), frameon=False)
    plt.plot(bandwidths, errs, '-', linewidth=3)
    plt.savefig('Tp1-NB.png', dpi=300)
    plt.show()
    plt.close()
    kdes, logp0, logp1 = fitNb(X_r,Y_r, best_bandwidth)
    return 1-scoreNb(kdes, logp0, logp1,X_t,Y_t), best_bandwidth, predictNb(kdes, logp0, logp1, X_t, Y_t)

def scoreNb(kdes,logp0,logp1,X,Y):
    p0 = np.ones(X.shape[0])*logp0
    p1 = np.ones(X.shape[0])*logp1
    for ix in range(X.shape[1]):
        p0 = p0  + kdes[ix][0].score_samples(X[:,[ix]])
        p1 = p1  + kdes[ix][1].score_samples(X[:,[ix]])
    pred = np.zeros(X.shape[0])
    pred[p1>p0] = 1
    return np.sum(pred==Y)/float(len(Y))

def predictNb(kdes,logp0,logp1,X,Y):
    p0 = np.ones(X.shape[0])*logp0
    p1 = np.ones(X.shape[0])*logp1
    for ix in range(X.shape[1]):
        p0 = p0  + kdes[ix][0].score_samples(X[:,[ix]])
        p1 = p1  + kdes[ix][1].score_samples(X[:,[ix]])
    pred = np.zeros(X.shape[0])
    pred[p1>p0] = 1
    return pred

def gscores(kf,X_r,Y_r,bw):
    va_score = []
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        kdes,logp0,logp1 = fitNb(X_r[tr_ix],Y_r[tr_ix],bw)
        va_score.append(scoreNb(kdes,logp0,logp1,X_r[va_ix],Y_r[va_ix]))
    return va_score

def fitNb(X,Y,bw):
    X0 = X[Y==0,:]
    X1 = X[Y==1,:]
    kdes = []
    logp0 = np.log(X0.shape[0]/float(X.shape[0]))
    logp1 = np.log(X1.shape[0]/float(X.shape[0]))
    for ix in range(X0.shape[1]):
        kde0 = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde0.fit(X0[:,[ix]])
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde1.fit(X1[:,[ix]])
        kdes.append((kde0,kde1))
    return kdes,logp0, logp1















        
#implementar Logistic, Kneighborns
