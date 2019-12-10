import os

import numpy as np
from pylab import bone, colorbar, pcolor, plot, show
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from utils.som.minisom import MiniSom


def reduct(X, n_components=3, seed=0):
    tsne = TSNE(n_components=n_components,
                learning_rate=300,
                perplexity=30,
                early_exaggeration=12,
                init="pca",
                random_state=seed)

    X_tsne = tsne.fit_transform(X)

    return X_tsne


def clusterize(X):
    dbscan = DBSCAN(eps=2, min_samples=5).fit(X)

    m = dbscan.labels_

    return m


def som(X, plot=True, seed=0):
    som = MiniSom(x=10,
                  y=10,
                  input_len=X.shape[1],
                  sigma=1.0,
                  learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)
    dmap = som.distance_map().T

    if plot:
        bone()
        pcolor(dmap)
        colorbar()
        show()

    mappings = som.win_map(X)

    return dmap, mappings
