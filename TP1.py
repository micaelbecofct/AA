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
from math import log
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
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
    Xs, Ys = get_data(filename)
    X_r, X_t, Y_r, Y_t = train_test_split(Xs, Ys, test_size = 0.33, stratify = Ys)
    folds = 5
    Kf = StratifiedKFold(Y_r, n_folds = folds)
    KnnErr, bestN, KnnPred = Knn(Kf, X_r, Y_r, X_t, Y_t) #KnnPred AA-07
    print("KnnErr, bestN", KnnErr, bestN)
    LogScore, bestC, LogPred = Logistic(Kf, X_r, Y_r, X_t, Y_t)
    print("LogisticScore, bestC", LogScore, bestC)
    NBScore, bestBandwidth, NBPred=NaiveBayes(Kf, X_r, Y_r, X_t, Y_t)
    print("NBScore, bestBandwidth", NBScore, bestBandwidth)
    MCNmarKnn_Log=MCNmar(KnnPred, LogPred, Y_t) #(|e01-e10|-1)²/e01+e10
    MCNmarNB_Log=MCNmar(NBPred, LogPred, Y_t)
    MCNmarNB_Knn=MCNmar(NBPred, KnnPred, Y_t)
    print("MCNmarKnn_Log", MCNmarKnn_Log)
    print("MCNmarKnn_Log", MCNmarNB_Log)
    print("MCNmarKnn_Log", MCNmarNB_Knn)
      
    
def Knn(Kf, X_r, Y_r, X_t, Y_t):
    N=1; Ns=[]; lowest=10000 ; errs=[]
    for ix in range (20): # 40/2 only odd values
        reg=KNeighborsClassifier(N) #Para o logistic aqui e LogisticRegression mas umas alteracoes
        scores = cross_val_score(reg, X_r, Y_r,cv=Kf)
        va_err = 1-np.mean(scores)
        if va_err < lowest:
            lowest= va_err; Best_n = N
        errs.append(va_err); Ns.append(N); N=N+2
    errs = np.array(errs)
    plt.figure(figsize=(8,8),frameon=False)
    plt.plot(Ns, errs,'-',linewidth=3)
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

def calc_fold(X,Y, Kf,C=1e12):
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
        va_err = calc_fold(X_r, Y_r, Kf,C=C )
        if va_err <= best_va:
            best_va = va_err; best_C = C
        errs.append(va_err)
        Cs.append(log(C))
        C*=2
    errs = np.array(errs)
    Cs = np.array(Cs)
    fig = plt.figure(figsize=(8,8),frameon=False)
    plt.plot(Cs,errs)
    fig.savefig('p_3.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    reg=LogisticRegression(C=best_C, tol=1e-10); reg.fit(X_r, Y_r)
    return 1-reg.score(X_t, Y_t), best_C, reg.predict(X_t)
    
def NaiveBayes (Kf, X_r, Y_r, X_t, Y_t):
    best_bandwidth=0.01
    lowest=10000 
    errs=[]
    for bandwidth in range(1,100,2):
        reg = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth/100)
        """
        e_log = np.log(float(e_train.shape[0])/tot_len) #isto mas com pontos de classe 0
        p_log = np.log(float(p_train.shape[0])/tot_len) #isto mas com pontos de classe 1
        c_e = classify(e_hists,e_log,p_hists,p_log,e_test) #classify devolve classe prevista
        c_p = classify(e_hists,e_log,p_hists,p_log,p_test)
        """
        #em vez de reg.predict devia ser uma especie de classify das teoricas
        acc_score = accuracy_score(Y_t, reg.predict(X_t), normalize=False)
        if acc_score < lowest:
            lowest= acc_score; best_bandwidth = bandwidth
        errs.append(acc_score)
    errs = np.array(errs)
    plt.figure(figsize=(8,8),frameon=False)
    plt.plot(range(1,100,2), errs,'-',linewidth=3)
    plt.show()
    plt.close()
    reg=KernelDensity(kernel='gaussian', bandwidth=best_bandwidth/100)
    return 1-reg.score(X_t, Y_t), best_bandwidth/100, reg.predict(X_t)
        
#implementar Logistic, Kneighborns
