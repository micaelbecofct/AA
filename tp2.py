#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gabriel Batista 47590
Micael Beco 48159
"""
"""
plano:
1 - obter dados do ficheiro
2 - fazer clustering com k-means
3 - fazer clustering com Gaussian Mixture Models
4 - fazer clustering com DBSCAN
4.1 - mapear cada ponto a sua distancia ao 4o vizinho mais proximo
4.2 - minPts vai ser numero de vizinhos no "cotovelo" do grafico(sempre 4?)
4.3 - epsilon vai ser distancia do "cotovelo" ao 4o vizinho

Internal
    silhouette score
External
    rand index
    precision
    recall
    f1 measure
"""

import pandas as pd
import numpy as np
from sklearn import cluster as cl
from plotClasses import plot_classes
RADIUS = 6371 #raio da terra em km

def lat_lon_to_3d(lat,lon):
    x = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    z = RADIUS * np.sin(lat * np.pi / 180)
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
    points = []
    for ix in range(len(latitudes)):
        x,y,z = lat_lon_to_3d(latitudes[ix],longitudes[ix])
        points.append([x,y,z])
    return points

#devolve array com clusterings para cada num de clusters
def k_means_cluster(points):
    clustering_labels = [];
    for ix in range(60,180):#ha 90 falhas        
        clustering = cl.KMeans(n_clusters = 10).fit(points)
        clustering_labels.append(clustering.labels_)
    return clustering_labels
    
faults, latitudes, longitudes = get_data("tp2_data.csv")
points = all_points_to_3d(latitudes, longitudes)
print("\npoints[5]: ",points[5],"\n")
k_means_cluster(points)













