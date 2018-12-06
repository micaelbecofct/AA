#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gabriel Batista 47590
Micael Beco 48159
"""

import pandas as pd
import numpy as np
from sklearn import cluster as cl
from sklearn import mixture as mx
from sklearn import neighbors as nb
from plotClasses import plot_classes
import matplotlib.pyplot as plt
from myaux import plot_performance, compute_silhouette, compute_rand_indexes

RADIUS = 6371  # raio da terra em km
EPSILON = 135


def lat_lon_to_3d(lat, lon):
    x = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = RADIUS * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    z = RADIUS * np.sin(lat * np.pi / 180)
    return x, y, z


# guardar latitude, longitude e falha de cada sismo
def get_data(filename):
    data = pd.read_csv(filename)
    faults = data["fault"]
    latitudes = data["latitude"]
    longitudes = data["longitude"]
    return faults, latitudes, longitudes


# devolve array com (x,y,z) de cada sismo
def all_points_to_3d(latitudes, longitudes):
    points = []
    for ix in range(len(latitudes)):
        x, y, z = lat_lon_to_3d(latitudes[ix], longitudes[ix])
        points.append([x, y, z])
    return points


# devolve array com clusterings para cada num de clusters
def k_means_cluster(points, k_clusters_range=range(10, 105, 5)):
    clustering_labels = []
    ks = []
    print("started clustering with k-means")
    for k in k_clusters_range:  # ha 90 falhas
        clustering = cl.KMeans(n_clusters=k).fit(points)
        clustering_labels.append(clustering.labels_)
        print(k, "clusters, ", clustering.n_iter_, "iterations this time")
        ks.append(k)
    return ks, clustering_labels


# devolve array com clusterings para cada num de componentes gaussianos
def gauss_mm(points, c_components_range=range(10, 105, 5)):
    clustering_labels = []
    cs = []
    print("started clustering with Gaussian Mixture Models")
    for c in c_components_range:  # ha 90 falhas
        clustering = mx.GaussianMixture(n_components=c).fit(points)
        clustering_labels.append(clustering.predict(points))
        print(c, "clusters, ", clustering.n_iter_, "iterations this time")
        cs.append(c)
    return cs, clustering_labels


def dbscan_labels(points, epsilon_range=range(EPSILON - 50, EPSILON + 100, 5)):
    clustering_labels = []
    print("started clustering with DBSCAN")
    es = []
    for e in epsilon_range:  # ha 90 falhas
        clustering = cl.DBSCAN(e).fit(points)
        clustering_labels.append(clustering.labels_)
        print("Neighbourhood radius: ", e)
        es.append(e)
    return es, clustering_labels


# devolve o epsilon ideal para o dbscan
def plot_for_dbscan(points, faults):
    dists = []
    knn = nb.KNeighborsClassifier(n_neighbors=4)
    knn.fit(points, np.zeros(len(points)))
    for ix in range(len(points)):
        curr = knn.kneighbors([points[ix]], n_neighbors=4)[0]
        dists.append(curr.item(3))  # o kneighbors devolve o array ordenado, a dist mais alta e a ultima
    dists.sort()
    dists.reverse()  # dists fica ordenado da distancia maior para a mais pequena
    plt.plot(dists, 'k,')
    plt.show()
    num_f = num_noise(faults)
    print("\nEpsilon ideal:", dists[num_f], "\nNum. de sismos sem falha:", num_f)
    return


# devolve o numero de pontos nao associados a uma falha
def num_noise(faults):
    count = 0
    for ix in range(1, len(faults)):
        if faults[ix] == -1:
            count = count + 1
    return count


# devolve array com clusterings para cada num de componentes gaussianos

def plot_kmeans(points, faults):
    ks, kmeans = k_means_cluster(points)
    print("Computing K-means performance")
    silhouette, precision, recall, rand, f1, adj_rand = compute_silhouette(kmeans, faults, points)
    print("Plotting K-means performance")
    plot_performance(ks, "Number of clusters", "Index Score", "./Kmeans.png",
                     silhouette, precision, recall, rand, f1, adj_rand)


def plot_gauss(points, faults):
    cs, gauss = gauss_mm(points)
    print("Computing Gaussian components performance")
    silhouette, precision, recall, rand, f1, adj_rand = compute_silhouette(gauss, faults, points)
    print("Plotting Gaussian components performance")
    plot_performance(cs, "Number of Gaussian components", "Index Score", "./Gauss.png",
                     silhouette, precision, recall, rand, f1, adj_rand)


def plot_dbscan(points, faults):
    epsilons, dbscan = dbscan_labels(points)
    print("Computing DBSCAN performance")
    silhouette, precision, recall, rand, f1, adj_rand = compute_silhouette(dbscan, faults, points)
    print("Plotting DBSCAN performance")
    plot_performance(epsilons, "Neighbourhood Distance", "Index Score", "./DBSCAN.png",
                     silhouette, precision, recall, rand, f1, adj_rand)


def show_plots():
    faults, latitudes, longitudes = get_data("tp2_data.csv")
    points = all_points_to_3d(latitudes, longitudes)
    plot_kmeans(points, faults)
    plot_gauss(points, faults)
    plot_dbscan(points, faults)
    plot_for_dbscan(points, faults)


def plot_dbscan_classes(epsilon, points, lon, lat):
    epsilons, dbscan = dbscan_labels(points, range(epsilon, epsilon+1))
    print(dbscan)
    plot_classes(dbscan[0], lon, lat)


def plot_kmeans_classes(k_clusters, points, lon, lat):
    ks, kmeans = k_means_cluster(points, range(k_clusters, k_clusters+1))
    plot_classes(kmeans, lon, lat)


def plot_gauss_classes(c_components, points, lon, lat):
    cs, gauss = gauss_mm(points, range(c_components,c_components+1))
    plot_classes(gauss, lon, lat)


def plot_all_classes(epsilon, c_components, k_clusters):
    faults, latitudes, longitudes = get_data("tp2_data.csv")
    points = all_points_to_3d(latitudes, longitudes)
    plot_dbscan_classes(epsilon, points, longitudes, latitudes)
    plot_kmeans_classes(k_clusters, points, longitudes, latitudes)
    plot_gauss_classes(c_components, points, longitudes, latitudes)


#show_plots()

# faults, latitudes, longitudes = get_data("tp2_data.csv")
# points = all_points_to_3d(latitudes, longitudes)
# plot_for_dbscan(points, faults)

# plot_classes(kmeans[0], longitudes, latitudes)
# print("gaussian[0]: ",gaussian[0])
# plot_classes(gaussian[0],longitudes,latitudes)
# print(dists)
