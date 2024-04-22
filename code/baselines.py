import numpy as np


def Uniform(X, k):
    # sample all points uniformly at random
    return np.random.choice(X.shape[0], k)


def FurthestFirst(X, k):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = np.random.choice(n, 1).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k-1):
        dist = np.array([np.linalg.norm(X-X[i], ord=2, axis=1) for i in ind])
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, np.arange(n)]
        # choose the point that is furthest away
        # from the points already chosen
        i = dist.argmax()
        ind.append(i)

    return ind


def FurthestSum(X, k):
    # Archetypal Analysis for Machine Learning
    # Morten Mørup and Lars Kai Hansen, 2010

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = np.random.choice(n, 1).item()
    ind.append(i)

    # compute the (sum) of distances to the chosen point(s)
    # dist = np.sum((X-X[i])**2, axis=1)
    dist = np.linalg.norm(X-X[i], ord=2, axis=1)
    initial_dist = dist.copy()

    # chose k-1 points
    for _ in range(k-1):
        # don't choose a chosen point again
        dist[ind] = 0.0
        # choose the point that is furthest away
        # to the sum of distances of points
        i = dist.argmax()
        ind.append(i)
        # add the distances to the new point to the current distances
        # dist = dist + np.sum((X-X[i])**2, axis=1)
        dist = dist + np.linalg.norm(X-X[i], ord=2, axis=1)

    # forget the first point chosen
    dist = dist - initial_dist
    ind = ind[1:]
    # don't choose a chosen point again
    dist[ind] = 0.0
    # chose another one
    i = dist.argmax()
    ind.append(i)

    return ind


def AAcoreset(X, k):
    # Coresets for Archetypal Analysis
    # Mair and Brefeld, 2019
    n = X.shape[0]
    dist = np.sum((X-X.mean(axis=0))**2, axis=1)
    q = dist / dist.sum()
    ind = np.random.choice(n, k, p=q)
    return ind

