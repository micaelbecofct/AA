"""
Gabriel Batista 47590
Micael Beco 48159
"""

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt


def plot_silhouette(points, clusterings_labels):
    s = np.zeros(len(clusterings_labels))
    for idx in range(len(clusterings_labels)):
        s[idx] = silhouette_score(points, clusterings_labels[idx])

    plt.plot(range(), s)
    plt.show()
