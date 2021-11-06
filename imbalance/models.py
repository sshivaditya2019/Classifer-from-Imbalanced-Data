'''Models for basic Data Clustestering and Analysis'''


from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as df
import numpy as np
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
        plt.scatter(df[label == i,0], df[label == i,1],label = i)
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