#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gabriel Batista 47590
Micael Beco 48159
"""
"""
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
from sklearn import mixture as mx
from sklearn import neighbors as nb
from plotClasses import plot_classes
import matplotlib.pyplot as plt
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
def a_to_3ll_pointsd(latitudes, longitudes):
	points = []
	for ix in range(len(latitudes)):
		x,y,z = lat_lon_to_3d(latitudes[ix],longitudes[ix])
		points.append([x,y,z])
	return points

#devolve array com clusterings para cada num de clusters
def k_means_cluster(points):
	clustering_labels = [];
	print("started clustering with k-means")
	for ix in range(10,105,5):#ha 90 falhas
		clustering = cl.KMeans(n_clusters = ix).fit(points)
		print(ix,"clusters, ",clustering.n_iter_,"iterations this time")
		clustering_labels.append(clustering.labels_)
	return clustering_labels

#devolve array com clusterings para cada num de componentes gaussianos
def gauss_mm(points):
	clustering_labels = [];
	print("started clustering with Gaussian Mixture Models")
	for ix in range(10,105,5):#ha 90 falhas
		clustering = mx.GaussianMixture(n_components = ix).fit(points)
		clustering_labels.append(clustering.predict(points))
		print(ix,"clusters, ",clustering.n_iter_,"iterations this time")
	return clustering_labels

def plot_for_dbscan(points,faults):
    dists = []
    knn = nb.KNeighborsClassifier(n_neighbors=4)
    knn.fit(points, np.zeros(len(points)))
    for ix in range(len(points)):
        curr = knn.kneighbors([points[ix]],n_neighbors=4)[0]
        dists.append(curr.item(3))#o kneighbors devolve o array ordenado, a dist mais alta e a ultima
    dists.sort(); dists.reverse()#dists fica ordenado da distancia maior para a mais pequena
    plt.plot(dists,'k,'); plt.show();
    num_f = num_noise(faults)
    print("\nEpsilon ideal:",dists[num_f],"\nNum. de sismos sem falha:",num_f)

def num_noise(faults):#devolve o numero de pontos nao associados a uma falha
    count = 0
    for ix in range(1,len(faults)):
        if(faults[ix] == -1):
            count = count + 1
    return count



faults, latitudes, longitudes = get_data("tp2_data.csv")
points = all_points_to_3d(latitudes,longitudes)
#plot_for_dbscan(points,faults)
#print("points[5]: ",points[5],"\n")
kmeans = k_means_cluster(points)
print("kmeans[0]: ",kmeans[0])
plot_classes(kmeans[0],longitudes,latitudes)
"""
gaussian = gauss_mm(points)
print("gaussian[0]: ",gaussian[0])
plot_classes(gaussian[0],longitudes,latitudes)
print(dists)
"""



























