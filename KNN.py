#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def Knn(Kf, X_r, Y_r, X_t, Y_t):
    N=1; Ns=[]; lowest=10000 ; errs=[]
    for ix in range (20): # 40/2 only odd values
        reg=KNeighborsClassifier(N) #Para o logistic aqui é LogisticRegression mas umas alteracoes
        scores = cross_val_score(reg, X_r, Y_r,cv=Kf)
        va_err = 1-np.mean(scores)
        if va_err < lowest:
            lowest= va_err; Best_n = N
        errs.append(va_err); Ns.append(N); N=N+2
    errs = np.array(errs)
    
    fig = plt.figure(figsize=(8,8),frameon=False)
    plt.title('K-Nearest Neighbours') 
    plt.ylabel('Error')
    plt.xlabel('Number of Neighbours')
    plt.plot(Ns, errs,'-',linewidth=3, label='Validation Error')
    plt.legend()
    fig.savefig('KNN.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    reg=KNeighborsClassifier(Best_n); reg.fit(X_r, Y_r)
    return 1-reg.score(X_t, Y_t), Best_n, reg.predict(X_t)
