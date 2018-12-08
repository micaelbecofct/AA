"""
Gabriel Batista 47590
Micael Beco 48159
"""

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt


def compute_scores(labels, faults, points):
    print("Computing silhouette scores")
    silhouettes = []
    precisions = []
    recalls = []
    rands = []
    f1s = []
    adj_rands = []
    best_silhouette = [-1, -1]
    best_precision = [-1, -1]
    best_recall = [-1, -1]
    best_rand = [-1, -1]
    best_f1 = [-1, -1]
    best_adj_rand = [-1, -1]
    
    for ix in range(len(labels)):
        silhouette = silhouette_score(points, labels[ix])
        silhouettes.append(silhouette)
        if best_silhouette[0] < silhouette: 
            best_silhouette[0] = silhouette
            best_silhouette[1] = ix
        precision, recall, rand, f1, adj_rand = compute_ext_indexes(labels[ix], faults)
        precisions.append(precision)
        if best_precision[0]< precision:
            best_precision[0] = precision
            best_precision[1] = ix
        recalls.append(recall)
        if best_recall[0] < recall :
            best_recall[0] = recall
            best_recall[1] = ix
        rands.append(rand)
        if best_rand[0] < rand : 
            best_rand[0] = rand
            best_rand[1] = ix
        f1s.append(f1)
        if best_f1[0] < f1 :
            best_f1[0] = f1
            best_f1[1] = ix
        adj_rands.append(adj_rand)
        if best_adj_rand[0] < adj_rand : 
            best_adj_rand[0] = adj_rand
            best_adj_rand[1]= ix   
    return silhouettes, precisions, recalls, rands, f1s, adj_rands, best_silhouette, best_precision, best_recall, best_rand, best_f1, best_adj_rand


def compute_ext_indexes(labels, faults):
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
    plt.plot(range, silhouette, color=[0.8500, 0.3250, 0.0980], linestyle='-', linewidth=1, label="Silhouette Score")
    plt.plot(range, rand, color=[0, 0.4470, 0.7410], linestyle='-', linewidth=1, label="Rand Score")
    plt.plot(range, precision, color=[0.9290, 0.6940, 0.1250], linestyle='-', linewidth=1, label="Precision")
    plt.plot(range, recall, color=[0.4940, 0.1840, 0.5560], linestyle='-', linewidth=1, label="Recall")
    plt.plot(range, f1, color=[0.6350, 0.0780, 0.1840], linestyle='-', linewidth=1, label="F1 Score")
    plt.plot(range, adj_rand, color=[0.3010, 0.7450, 0.9330], linestyle='-', linewidth=1, label="Adjusted Rand Score")
    plt.legend()
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
