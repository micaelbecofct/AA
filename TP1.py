#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Micael Beco
@author: Gabriel Baptista
"""
import numpy as np
from Logistic import Logistic
from KNN import Knn
from NaiveBayes import NaiveBayes
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
#from sklearn.metrics import accuracy_score
def get_data(filename):
    mat = np.loadtxt(filename, delimiter = ',')
    
    data = shuffle(mat)
    Ys = data[:,-1] #-1 e a ultima coluna
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
    
    NBScore, bestBandwidth, NBPred=  NaiveBayes(Kf, X_r, Y_r, X_t, Y_t)
    print("NBScore, bestBandwidth", NBScore, bestBandwidth)
    
    MCNmarKnn_Log=MCNmar(KnnPred, LogPred, Y_t) #(|e01-e10|-1)²/e01+e10
    MCNmarNB_Log=MCNmar(NBPred,LogPred, Y_t)
    MCNmarNB_Knn=MCNmar(KnnPred,NBPred, Y_t)
    print("MCNmarKnn_Log", MCNmarKnn_Log)
    print("MCNmarKB_Log", MCNmarNB_Log)
    print("MCNmarKB_Knn", MCNmarNB_Knn)
      
    
    
    
def MCNmar(PredA, PredB,y):
    TrueA = PredA==y 
    FalseB = PredB !=y
    TrueB = PredB == y
    FalseA = PredA != y
    NTaFb = sum(TrueA*FalseB)
    NTbFa = sum(TrueB*FalseA)
    return ((abs(NTaFb-NTbFa)-1)**2)*1.0/(NTaFb+NTbFa)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
