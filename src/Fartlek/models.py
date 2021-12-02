'''Models for basic Data Clustestering and Analysis'''


import pandas as pd
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import cluster
from scipy.spatial.distance import pdist
import numpy as np
from scipy.cluster.hierarchy import cut_tree
from .result_evaluate import confuseResults

def PCA_Decompsition(
    data
):
    pcs = PCA(2)
    df = pcs.fit_transform(data)
    return df

def KMeans_Cluster_Plot(
    df_dec,
    n_class = 2
):
    kmeans = KMeans(n_clusters = 2)
    label = kmeans.fit_predict(df_dec)
    u_labels = np.unique(label)
    centroids = kmeans.cluster_centers_
    for i in u_labels:
        plt.scatter(df_dec[label == i,0], df_dec[label == i,1],label = i)
    plt.scatter(centroids[:,0], centroids[:,1], s = 80)
    plt.legend()
    plt.show()

def KNeigbourClassResults(
    X,
    y
):
    neigh = KNeighborsClassifier(n_neighbors = 2)
    neigh.fit(X,y)
    confuseResults(y,neigh.predict(X))

def AggloClust(
    X,
    y,
    data
):
    Z = cluster.hierarchy.ward(pdist(X))
    cut = cut_tree(Z, n_clusters=2)
    labels = list([i[0] for i in cut])
    fig, axes = plt.subplot(nrow = 1, ncols = 1, figsize = (4,4), dpi = 300)
    plot_tree(cut, feature_names=labels, filled=True, class_names=True)

