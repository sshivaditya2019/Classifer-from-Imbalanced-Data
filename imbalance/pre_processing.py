'''Preprocessing the data to improve the perfomence of classifier'''

'''
Inorder to imporve the perfomence we are going to use 4 resampling methods
* Simple Random over-sampling
* Cluster based over-sampling
* Neighbourhood Cleaning Rule(NCR)
* SPIDER

Apart from these we are also going to implement and use SMOTE(Synthetic Minority Over Sampling)
'''

from collections import Counter
import numpy as np
from sklearn import cluster
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from clover.over_sampling import ClusterOverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
from multi_imbalance.resampling.spider import SPIDER3
from sklearn.decomposition import PCA 
from collections import Counter

def extract_characterstics(
    X, 
    y
):
    n_samples, n_features = X.shape
    count_y = Counter(y)
    (maj_label, n_samples_maj), (min_label, n_samples_min) = count_y.most_common()
    ir = n_samples_maj / n_samples_min
    return n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min, ir

def print_characterstics(
    X,
    y
):
    n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min, ir = extract_characterstics(X, y)
    print(
        f'Number of samples: {n_samples}',
        f'Number of features: {n_features}',
        f'Majority class label: {maj_label}',
        f'Number of majority class samples: {n_samples_maj}',
        f'Minority class label: {min_label}',
        f'Number of Minority class sample: {n_samples_min}',
        f'Imbalance Ratio : {ir:.1f}',
        sep='\n'
    )

rnd_seed = 12

def RandomOverSampling(
    X,
    y
):
    ros = RandomOverSampler(random_state = 42)
    X_res, y_res = ros.fit_resample(X, y)
    print("Original Data Characterstics")
    print_characterstics(X,y)
    print("Data Characterstics after random Over Sampling")
    print_characterstics(X_res,y_res)
    return (X_res, y_res)


def ClusterBasedOverSampling(
    X,
    y
):
    smote = SMOTE(random_state = rnd_seed + 1)
    kmeans = KMeans(n_clusters=2, random_state=rnd_seed+5)
    kmeans_smote = ClusterOverSampler(oversampler = smote, clusterer = kmeans)
    X_res, y_res= kmeans_smote.fit_resample(X,y)
    print("Original Data Characterstics")
    print_characterstics(X,y)
    print("Data Characterstics after Cluster Based Over Sampling")
    print_characterstics(X_res,y_res)
    return (X_res, y_res)

def NCR(
    X,
    y
):
    ncr = NeighbourhoodCleaningRule()
    X_res, y_res = ncr.fit_resample(X, y)
    print("Original Data Characterstics")
    print_characterstics(X,y)
    print("Data Characterstics after NCR")
    print_characterstics(X_res,y_res)
    return (X_res, y_res)

def SPIDER(
    X,
    y
):
    cost = np.random.rand(4).reshape((2,2))
    for i in range(2):
        cost[i][i] = 0
    n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min, ir = extract_characterstics(X,y)
    maj_int_min = {
        'maj':[maj_label],
        'int':[],
        'min':[min_label]
    }
    clf = SPIDER3(k=1,maj_int_min=maj_int_min,cost = cost)
    X_res, y_res = clf.fit_resample(X, y)
    print("Original Data Characterstics")
    print_characterstics(X,y)
    print("Data Characterstics after SPIDER")
    print_characterstics(X_res,y_res)
    return (X_res, y_res)
