#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from math import log
from sklearn.cross_validation import cross_val_score



def calc_validation_error(X,Y, Kf,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    scores = cross_val_score(reg, X, Y,cv=Kf)
    va_err = 1-np.mean(scores)
    return va_err

def Logistic(Kf, X_r, Y_r, X_t, Y_t):
    best_C=1
    errs = []
    C=1
    Cs=[]
    best_va= 10000
    """Generate folds and loop"""
    for ic in range(20):
        va_err = calc_validation_error(X_r, Y_r, Kf,C=C )
        if va_err <= best_va:
            best_va = va_err; best_C = C
        errs.append(va_err)
        Cs.append(log(C))
        C*=2
    errs = np.array(errs)
    Cs = np.array(Cs)
    
    fig = plt.figure(figsize=(8,8),frameon=False)
    plt.title('Logistic Regression') 
    plt.ylabel('Error')
    plt.xlabel('log(C)')
    plt.plot(Cs,errs,'-',linewidth=3,label='Validation Error')
    plt.legend()
    fig.savefig('Logistic.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    reg=LogisticRegression(C=best_C, tol=1e-10); reg.fit(X_r, Y_r)
    return 1-reg.score(X_t, Y_t), best_C, reg.predict(X_t)
    
