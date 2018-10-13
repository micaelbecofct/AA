#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:09:21 2018

@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity

def get_data(filename):
    mat = np.loadtxt(filename, delimiter = ',')
    data = shuffle(mat)
    Ys = data[:,-1]
    Xs = data[:,:-1]
    means = np.mean(Xs,axis=0) #media
    stdevs = np.std(Xs,axis=0) #desvio padrao
    Xs = (Xs-means)/stdevs
    return Xs, Ys
    
def compare(filename): #filename vai ser Tp1_data.csv
    Xs, Ys = get_data(Filename)
    X_r, X_t, Y_r, Y_t = train_test_split(Xs, Ys, test_size = 0.33, Stratify = Ys)
    folds = 5
    Kf = StratifiedKFold(Y_r, n_folds = folds)
    KnnErr, bestN, KnnPred = Knn(Kf, X_r, Y_r, X_t, Y_t) #KnnPred AA-07
    print("KnnErr, bestN", KnnErr, bestN)
    LogScore, LogPred = Logistic(Kf, X_r, Y_r, X_t, Y_t)
    print("LogisticScore", LogScore)
    #BASE ...
    MCNmarKnnprog=MCNmar(KnnPred, LogPred, Y_t) #(|e01-e10|-1)²/e01+e10
    
def Knn(Kf, X_r, Y_r, X_t, Y_t):
    N=1; Ns=[]; lowest=10000 ; errs=[]
    for ix in range (20): # 40/2 only odd values
        reg=KNeighborsClassifier(N) #Para o logistic aqui é LogisticRegression mas umas alteracoes
        scores = cross_val_score(reg, X_r, Y_r,cv=Kf)
        va_err = 1-np.mean(scores)
        if va_err < lowest:
            lowest= va_err; Best_n =N
        errs.append(va_err); Ns.append(N); N=N+2
    errs = np.array(errs)
    plt.figure(figsize=(8,8),frameon=False)
    plt.plot(Ns, errs,'-',linewigth=3)
    plt.show()
    plt.close()
    reg=KNeighborsClassifier(Best_n); reg.fit(X_r, Y_r)
    return 1-reg.score(X_t, Y_t), Best_n, reg.predict(X_t)

def MCNmar(PredA, PredB,y):
    TrueA = PredA==y 
    FalseB = PredB !=y
    TrueB = PredB == y
    FalseA = PredA != y
    NTaFb = sum(TrueA*FalseB)
    NTbFa = sum(TrueB*FalseA)
    return ((abs(NTaFb-NTbFa)-1)**2)*1.0/(NTaFb+NTbFa)

def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]    
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

def Logistic(Kf, X_r, Y_r, X_t, Y_t):
    best_C=1
    best_feats=2
    best_va= 10000
    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    errs = []
    C=1
    """Generate folds and loop"""
    for ic in range(20):
        best_va_c= 10000
        for feats in range(2,17):
            tr_err=va_err=0
            for tr_ix, va_ix in kf.split(Y_r, Y_r):  #Y_r vetor de classes para treino
                r,v = calc_fold(feats, X_r, Y_r, tr_ix, va_ix, C=C )
                tr_err += r
                va_err += v
            if va_err/folds <= best_va_c:
                best_va_c = va_err/folds
                if va_err/folds <= best_va:
                    best_va = va_err/folds; best_feats=feats ; best_C = C
            print (feats,':',tr_err/folds, va_err/folds)
        errs.append(best_va_c)
        C*=2
    errs = np.array(errs)
    fig = plt.figure(figsize=(8,8),frameon=False)
    plt.plot(2**range(20),errs[:],'-b', linewidth= 3)
    fig.savefig('p_3.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    reg=LogisticRegression(C=best_C, tol=1e-10); reg.fit(X_r[:best_feats], Y_r[:best_feats])
    return 
    #...LogisticRegression com base no Knn
    
    
    
#implementar Logistic, Kneighborns