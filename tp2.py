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
"""

import pandas as pd
import numpy as np
from sklearn import cluster as cl
from plotClasses import plot_classes
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
    points = []
    for ix in range(len(latitudes)):
        x,y,z = lat_lon_to_3d(latitudes[ix],longitudes[ix])
        curr = [x,y,z]
        points.append(curr)
    return points
        #fazer matriz com 3 colunas e todos os pontos

