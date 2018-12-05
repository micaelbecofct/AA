"""
Gabriel Batista 47590
Micael Beco 48159
"""

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt


def compute_silhouette(labels, faults, points):
    silhouettes = []
    precisions = []
    recalls = []
    rands = []
    f1s = []
    adj_rands = []

    for ix in range(len(labels)):
        silhouettes.append(silhouette_score(points, labels[ix]))
        precision, recall, rand, f1, adj_rand = compute_rand_indexes(labels[ix], faults)
        precisions.append(precision)
        recalls.append(recall)
        rands.append(rand)
        f1s.append(f1)
        adj_rands.append(adj_rand)
    return silhouettes, precisions, recalls, rands, f1s, adj_rands


def compute_rand_indexes(labels, faults):
    sf = sc = tp = tn = 0.0
    for ix in range(len(labels)):
        same_fault = faults[ix] == faults[ix + 1:]
        same_cluster = labels[ix] == labels[ix + 1:]
        sf += np.sum(same_fault)
        sc += np.sum(same_cluster)
        tp += np.sum(np.logical_and(same_fault, same_cluster))
        tn += np.sum(np.logical_and(np.logical_not(same_fault), np.logical_not(same_cluster)))
    total = len(labels) * (len(labels) - 1) / 2
    precision = tp / sc
    recall = tp / sf
    return precision, recall, (tp + tn) / total, precision * recall * 2 / (precision + recall), adjusted_rand_score(labels, faults)


def plot_performance(range, x_label, y_label, filename, silhouette, precision, recall, rand, f1, adj_rand):
    n_x_values = len(range)
    x_values_range = np.arange(n_x_values)
    plt.xticks(x_values_range, range)
    plt.plot(x_values_range, silhouette, color=[0.8500, 0.3250, 0.0980], linestyle='-', linewidth=1, label="Silhouette Score")
    plt.plot(x_values_range, rand, color=[0, 0.4470, 0.7410], linestyle='-', linewidth=1, label="Rand Score")
    plt.plot(x_values_range, precision, color=[0.9290, 0.6940, 0.1250], linestyle='-', linewidth=1, label="Precision")
    plt.plot(x_values_range, recall, color=[0.4940, 0.1840, 0.5560], linestyle='-', linewidth=1, label="Recall")
    plt.plot(x_values_range, f1, color=[0.6350, 0.0780, 0.1840], linestyle='-', linewidth=1, label="F1 Score")
    plt.plot(x_values_range, adj_rand, color=[0.3010, 0.7450, 0.9330], linestyle='-', linewidth=1, label="Adjusted Rand Score")
    plt.legend()
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
