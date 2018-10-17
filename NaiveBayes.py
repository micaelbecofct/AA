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
from sklearn.model_selection import GridSearchCV


def separateByClass(X, Y):
	separated = {}
	for i in range(len(X)):
		if (Y[i] not in separated):
			separated[Y[i]] = []
		separated[Y[i]].append(X[i])
	return separated

def classify(real_r,real_log,fake_r,fake_log,feat_mat, reg):
    classes = np.zeros(len(feat_mat))
    for row in range(len(feat_mat)):
        reg.fit(real_r)
        real_sum = real_log + reg.score(feat_mat)
        reg.fit(fake_r)
        fake_sum = fake_log + reg.score(feat_mat)
        if(real_sum < fake_sum):
            classes[row]= 1
    return classes
        
def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):
    separated_r = separateByClass(X_r,Y_r);
    separated_t = separateByClass(X_t,Y_t);
    real_r = separated_r[0]
    fake_r = separated_r[1]
    real_t = separated_t[0]
    fake_t = separated_t[1]
    best_bandwidth=1
    best_score = 0
    bandwidth = 20
    tot_len = len(real_r)+len(fake_r)
    real_log = np.log(len(real_r)/tot_len)
    fake_log = np.log(len(fake_r)/tot_len)
    print(real_r[:])
    for band in range(1,100,2):
         reg=KernelDensity(kernel='gaussian', bandwidth=band/100)
         #c_real = classify(real_r,real_log,fake_r, fake_log, real_t,reg)
         #c_fake = classify(real_r,real_log,fake_r, fake_log, fake_t,reg)
         #errors = sum(c_real)+sum(1-c_fake)



    """
    best_bandwidth=0.01 #width of the kernel
    lowest=10000 
    errs=[]
    bds = []
    for bandwidth in range(1,100,2):
        reg=KernelDensity(kernel='gaussian', bandwidth=bandwidth/100)
        reg.fit(X_r)
        reg.score_samples(arrayshape)
        va = 1-np.mean(scores)
        print(va)
    plt.figure(figsize=(8,8),frameon=False)
    plt.plot(bds, errs,'-',linewidth=3)
    plt.show()
    plt.close()
    reg=KernelDensity(kernel='gaussian', bandwidth=best_bandwidth/100); reg.fit(X_r,Y_r)
    return 1-reg.score(X_t, Y_t), best_bandwidth/100
   """
    return 0, 0

def scoreNB(kdes,logp0,logp1,X,Y):
    p0 = np.ones(X.shape[0])*logp0
    p1 = np.ones(X.shape[0])*logp1
    for ix in range(X.shape[1]):
        p0 = p0  + kdes[ix][0].score_samples(X[:,[ix]])
        p1 = p1  + kdes[ix][1].score_samples(X[:,[ix]])
    pred = np.zeros(X.shape[0])
    pred(p1>p0) = 1
    return np.sum(pred==Y)/float(len(Y))

def predictNB(kdes,logp0,logp1,X,Y):
    p0 = np.ones(X.shape[0])*logp0
    p1 = np.ones(X.shape[0])*logp1
    for ix in range(X.shape[1]):
        p0 = p0  + kdes[ix][0].score_samples(X[:,[ix]])
        p1 = p1  + kdes[ix][1].score_samples(X[:,[ix]])
    pred = np.zeros(X.shape[0])
    pred(p1>p0) = 1
    return pred

def gscores(kf,X_r,Y_r,bw):
    va_score = []
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        kdes,logp0,logp1 = fitNb(X_r[tr_ix],Y_r[tr_ix],bw)
        va_score.append(scoreNb(kdes,logp0,logp1,X_r[va_ix],Y_r[va_ix]))
    return va_score




















        
#implementar Logistic, Kneighborns
