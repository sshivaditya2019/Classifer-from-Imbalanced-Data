'''Data generation for simulation of class imbalance'''

import numpy as np
import numbers
import pandas as pd
from collections.abc import Iterable
from sklearn.utils import check_array
import unittest

def make_blobs(
    n_samples=100,
    n_features=2,
    *,
    centers=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=None,
    return_centers=False,
):
    generator = check_random_state(random_state)
    if isinstance(n_samples, numbers.Integral):

        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )

        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:

        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        try:
            assert len(centers) == n_centers
        except TypeError as e:
            raise ValueError(
                "Parameter `centers` must be array-like. Got {!r} instead".format(
                    centers
                )
            ) from e
        except AssertionError as e:
            raise ValueError(
                "Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            ) from e
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]

    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    X = []
    y = []

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(generator.normal(loc=centers[i], scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]

    if return_centers:
        return X, y, centers
    else:
        return X, y



def check_random_state(seed):

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def center_gen(
    n_samp = 300,
    x_o = 10,
    y_o =10 ,
    rs = 2,
    item = 5,
    lisRe = False
):
    
    
    n_samples = n_samp
    n_bins = 2
    x_0 = x_o
    y_0 = y_o
    r = rs
    items = item
    centers = []
    for i in range(items):
        x = x_0 + r*np.cos(2*np.pi * i/items)
        y = x_0 + r*np.sin(2*np.pi * i/items)
        centers.append((x,y))
    if lisRe:
        return centers
    else:
        return np.array(centers)


def gen_clover5(
    n_samples = 400,
    ratio = 0.5
):
    centers = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = 5)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.1, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=0.5,size=[n_samples]))

def gen_colver3(
    n_samples = 400,
    ratio = 0.5
):
    centers = center_gen(n_samp = 400,x_o = 10,y_o =10 ,rs = 1,item = 3)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.6, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples]))

def gen_pawdata02b(
    n_samples = 400,
    ratio = 0.5
):
    centers = center_gen(n_samp = 400,x_o = 10,y_o =10 ,rs = 1,item = 3,lisRe = True)
    centers += center_gen(n_samp = 1000,x_o = 12,y_o =12 ,rs = 2,item = 2,lisRe = True)
    centers = np.array(centers)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.5, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples]))

def gen_pawdata02a(
    n_samples = 400,
    ratio = 0.5
):
    centers = [(10,10),(10,15),(10,5)]
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.5, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples])) 

def gen_subclass3(
    n_samples = 400,
    ratio= 0.5
):
    centers = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = 6)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.1, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples])) 

def gen_subclass5(
    n_samples = 400,
    ratio = 0.5
):
    centers = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = 10)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.1, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples]))

def arrRatioMaintain1(X,ratio):
    X = np.random.binomial(n=1,p=ratio,size=[len(X)])
    return X
def arrRatioMaintain2(X,ratio):
    X = np.random.binomial(n=1,p=(1-ratio),size=[len(X)])
    return X
def constadd(X,alt):
    if ((alt >= 1) or (alt <= -1)):
        raise ValueError(
            "The Rare Example Generator does not work well with value"
            "of increase/decrease greater than 1 or less than -1. Got Value = {} "
           .format(alt)
        )
    for i in range(0,len(X)):
        X[i] += alt
    return X

    

def gen_overlapping(
    n_subclus = 3,
    n_samples = 400,
    ratio = 0.5
):
    centers1 = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = n_subclus)
    X1,_  = make_blobs(n_samples = n_samples//2, n_features = 2, cluster_std = 0.1, centers = centers1, shuffle=True, random_state=42)
    centers2 = center_gen(n_samp = 300//2,x_o = 10,y_o =10 ,rs = 0.5,item = n_subclus)
    X2,_  = make_blobs(n_samples = n_samples//2, n_features = 2, cluster_std = 0.1, centers = centers2, shuffle=True, random_state=42)
    Y1 = np.random.binomial(n=1,p=ratio,size=[n_samples//2])
    Y2 = np.random.binomial(n=1,p=ratio,size=[n_samples//2])
    Y1 = np.concatenate((Y1[:(n_samples//2)],arrRatioMaintain1(Y1[(n_samples//2)::],ratio)),axis = 0)
    Y2 = np.concatenate((Y2[:(n_samples//2)],arrRatioMaintain2(Y1[(n_samples//2)::],ratio)), axis = 0)
    X = np.concatenate((X1,X2), axis = 0 )
    y = np.concatenate((Y1,Y2), axis = 0 )
    return (X,y)

def gen_subclus(
    n_samples = 400,
    ratio = 0.5,
    n_clus = 3
):
    centers = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = n_clus*2)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.1, centers = centers, shuffle=True, random_state=42)
    return (X,np.random.binomial(n=1,p=ratio,size=[n_samples]))

def gen_rareborder(
    n_clus = 3,
    n_samples = 400,
    ratio = 0.5,
    rs = 0.5,
    val = 1
):
    centers = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = rs,item = n_clus*2)
    X,_  = make_blobs(n_samples = n_samples, n_features = 2, cluster_std = 0.1, centers = centers, shuffle=True, random_state=42)
    Y = np.random.binomial(n=1,p=ratio,size=[n_samples])
    y = np.concatenate((Y[:(n_samples//2)],constadd(Y[(n_samples//2)::],val)),axis = 0)
    return (X,y)

def gen_overlappingRareBorder(
    n_subclus = 3,
    n_samples = 400,
    ratio = 0.5,
    val = 0.99
):
    centers1 = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = n_subclus)
    X1,_  = make_blobs(n_samples = n_samples//2, n_features = 2, cluster_std = 0.1, centers = centers1, shuffle=True, random_state=42)
    centers2 = center_gen(n_samp = 300,x_o = 10,y_o =10 ,rs = 0.5,item = n_subclus)
    X2,_  = make_blobs(n_samples = n_samples//2, n_features = 2, cluster_std = 0.1, centers = centers2, shuffle=True, random_state=42)
    Y1 = np.random.binomial(n=1,p=ratio,size=[n_samples//2])
    Y2 = np.random.binomial(n=1,p=ratio,size=[n_samples//2])
    Y1 = np.concatenate((Y1[:(n_samples//2)],arrRatioMaintain1(Y1[(n_samples//2)::],ratio)),axis = 0)
    Y2 = np.concatenate((Y2[:(n_samples//2)],arrRatioMaintain2(Y1[(n_samples//2)::],ratio)), axis = 0)
    X = np.concatenate((X1,X2), axis = 0 )
    Y = np.concatenate((Y1,Y2), axis = 0 )
    y = np.concatenate((Y[:(n_samples//2)],constadd(Y[(n_samples//2)::],val)),axis = 0)
    return (X,y)

if __name__ == '__main__':
    X,y = make_blobs()
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)