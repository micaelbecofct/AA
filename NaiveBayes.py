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
from sklearn.model_selection import StratifiedKFold


	


        
def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):

    real_r = X_r[Y_r ==0]
    fake_r = X_r[Y_r ==1]
    real_t = X_r[Y_r ==0]
    fake_t = X_r[Y_r ==1]
    kf = StratifiedKFold
    best_bandwidth=1
    best_score = 0
    bandwidth = 20
    tot_len = len(real_r)+len(fake_r)
    real_log = np.log(len(real_r)/tot_len)
    fake_log = np.log(len(fake_r)/tot_len)
    #for band in range(1,100,2):
         #reg=KernelDensity(kernel='gaussian', bandwidth=band/100)
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

        
#implementar Logistic, Kneighborns