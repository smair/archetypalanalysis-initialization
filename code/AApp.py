import numpy as np
import archetypalanalysis as AA


def AApp(X, k, M=1000.0):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = np.random.choice(n, 1).item()
    ind.append(i)

    # sample second point
    dist = np.sum((X-X[i])**2, axis=1)
    i = np.random.choice(n, 1, p=dist/dist.sum()).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k-2):
        A = AA.ArchetypalAnalysis_compute_A(X, X[ind], M=M)
        dist = np.sum((X-A@X[ind])**2, axis=1)
        i = np.random.choice(n, 1, p=dist/dist.sum()).item()
        ind.append(i)

    return ind


# first approximation to AA++
# idea: approximate the distance computation
def k_means_pp(X, k):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = np.random.choice(n, 1).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k-1):
        dist = np.array([np.sum((X-X[i])**2, axis=1) for i in ind])
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, np.arange(n)]
        i = np.random.choice(n, 1, p=dist/dist.sum()).item()
        ind.append(i)

    return ind


# second approximation to AA++
# idea: approximate the sampling procedure
def AApp_MC(X, k, m, M=1000.0):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = np.random.choice(n, 1).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k-1):
        indices = np.random.choice(n, m)
        A = AA.ArchetypalAnalysis_compute_A(X[indices], X[ind], M=M)
        dist = np.sum((X[indices]-A@X[ind])**2, axis=1)
        i = indices[0]
        dist_i = dist[0]
        for j, dist_j in zip(indices, dist):
            if dist_i == 0.0 or dist_j/dist_i > np.random.rand():
                i = j
                dist_i = dist_j
        ind.append(i)

    return ind

