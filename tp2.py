#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gabriel Batista 47590
Micael Beco 48159
"""
"""plano:
1 - obter dados do ficheiro
2 - fazer clustering com k-means
3 - fazer clustering com Gaussian Mixture Models
4 - fazer clustering com DBSCAN
4.1 - mapear cada ponto a sua distancia ao 4o vizinho mais proximo
4.2 - minPts vai ser numero de vizinhos no "cotovelo" do grafico(sempre 4?)
4.3 - epsilon vai ser distancia do "cotovelo" ao 4o vizinho"""

import pandas as pd
import numpy as np
import matplotlib as mat
from sklearn import cluster as cl
RADIUS = 6371 #raio da terra em km

def lat_lon_to_3d(lat,lon):
    x = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    z = RADIUS * np.sen(lat * np.pi / 180)
    return x,y,z

#guardar latitude, longitude e falha de cada sismo
def get_data(filename):
    data = pd.read_csv(filename)
    faults = data["fault"]
    latitudes = data["latitude"]
    longitudes = data["longitude"]
    return faults, latitudes, longitudes

#devolve array com (x,y,z) de cada sismo
def all_points_to_3d(latitudes, longitudes):
    xs = []; ys = []; zs = [];
    for ix in range(len(latitudes)):
        x,y,z = lat_lon_to_3d(latitudes[ix],longitudes[ix])
        xs.append[x]; ys.append[y]; zs.append[z];
        #fazer matriz com 3 colunas e todos os pontos
        
def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = mat.imread("Mollweide_projection_SW.jpg")        
    mat.plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = mat.plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    mat.plt.close()
    mat.plt.figure(figsize=(10,5),frameon=False)    
    mat.plt.subplot(111)
    mat.plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        mat.plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        mat.plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    mat.plt.axis('off')
