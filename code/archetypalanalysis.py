import time
import numpy as np
from tqdm import tqdm

try:
    # for some datasets scipy.optimize.nnls fails because it runs out of iterations
    # in this version of nnls we modified the maximum number of iterations
    # the rest is untouched and identical to the scipy version
    from nnls import nnls
except ImportError:
    print('importing nnls failed; did you run ./build.sh ?')
    print('using nnls from scipy.optimize')
    from scipy.optimize import nnls


# residual sum of squares, given X, A, Z
def RSS_Z(X, A, Z):
    # RSS(k) = || X - AZ ||_F^2
    tmp = X - np.dot(A, Z)
    return np.sum(tmp ** 2)


def ArchetypalAnalysis_compute_A(X, Z, M=1000.0):
    # initialization
    n = X.shape[0]
    k = Z.shape[0]
    A = np.zeros((n, k))

    # || Z^t ai - xi ||^2
    # set up optimization of ai,
    # i.e., the convex combination for each data point xi
    Q = np.vstack((Z.T, M * np.ones(k)))
    for i in range(n):
        ai, rnorm = nnls(Q, np.hstack((X[i], M)))
        A[i] = ai.T

    return A


def ArchetypalAnalysis(X, Z, k, max_iterations=250, stop=True, epsilon=1e-3, M=1000.0, verbose=False):
    # initialization
    n = X.shape[0]
    A = np.zeros((n, k))  # convex combination for each data point xi, i=1..n
    B = np.zeros((k, n))  # convex combination for each archetype  zj, j=1..k

    iteration = 0
    rss = [-999]  # will be removed before returning
    time_after_iter = [time.time()]

    Q = np.vstack((X.T, M * np.ones(n)))
    for iteration in tqdm(range(1, max_iterations + 1), desc="AA"):
        # optimization of all ai's,
        # i.e., the convex combination for each data point xi
        A = ArchetypalAnalysis_compute_A(X, Z, M)

        # update (intermediate) archetypes
        # X = A Z
        # A^t X = A^t A Z
        # ( A^t A )^-1 A^t X = Z
        # Z = np.linalg.solve( np.dot( A.T, A ), np.dot( A.T, X ) )
        Z = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, X), rcond=None)[0]
        # Z = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, X))[0]
        # Z = np.dot( np.dot( np.linalg.inv( np.dot( A.T, A ) ), A.T ), X )

        # optimization of all bj's,
        # i.e. the convex combination for each archetype zj
        for j in range(k):
            b, rnorm = nnls(Q, np.hstack((Z[j], M)))
            B[j] = b.T

        # update archetypes
        Z = np.dot(B, X)

        # compute new RSS and store it
        rss.append(RSS_Z(X, A, Z))

        if verbose:
            print("Iteration %2d // RSS=%.3f" % (iteration, rss[-1]))

        # stop conditions
        converged = np.abs(rss[-1] - rss[-2]) / np.abs(rss[-1]) < epsilon
        increasing = rss[-1] > rss[-2] and len(rss) > 2
        outOfIter = iteration >= max_iterations
        if verbose and increasing:
            print("RSS is increasing; possibly not converged/optimal")
        if verbose and outOfIter and not converged:
            print("max. iterations reached; possibly not converged")

        time_after_iter.append(time.time()-time_after_iter[0])

        # test for stopping
        if stop and (converged or increasing or outOfIter):
            break

    A = ArchetypalAnalysis_compute_A(X, Z, M)

    return Z, A, B, rss[1:], time_after_iter[1:]

