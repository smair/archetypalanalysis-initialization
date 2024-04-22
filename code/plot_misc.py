import numpy as np
import matplotlib.pylab as plt
import matplotlib
import matplotlib.patches
import shapely.geometry

from scipy.spatial import ConvexHull

import AApp
import baselines
import archetypalanalysis as AA

from nnls import nnls

from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 12})

################################################################################
# AA

np.random.seed(23)
X1 = np.random.multivariate_normal([-3,-1], np.eye(2), 50)
X2 = np.random.multivariate_normal([+3,-1], np.eye(2), 50)
X3 = np.random.multivariate_normal([+0,+6], np.eye(2), 50)
X4 = np.random.multivariate_normal([+0,+1], np.eye(2), 50)
X5 = np.random.multivariate_normal([+0,+3], np.eye(2), 50)

X = np.vstack((X1,X2,X3,X4,X5))


hullX = ConvexHull(X)
k = 4
col = 'royalblue'

np.random.seed(0)
ind = AApp.AApp(X, k)
Z_init = X[ind].copy()

M = 1000.0
Z, A, B, rss_AA, time_AA = AA.ArchetypalAnalysis(X, Z_init, k, stop=True, max_iterations=100, M=M)


fig, ax = plt.subplots(figsize=(6,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.add_artist(matplotlib.patches.Polygon(X[hullX.vertices], color='black', alpha=0.05, label=r'Convex Hull of $\mathbf{X}$'))
hullZ = ConvexHull(Z)
P = shapely.geometry.Polygon(Z).convex_hull
ax.add_artist(matplotlib.patches.Polygon(Z[hullZ.vertices], color=col, alpha=0.1, label=r'Convex Hull of $\mathbf{Z}$') )
for simplex in hullZ.simplices:
    ax.plot(Z[simplex,0], Z[simplex,1], 'k-',alpha=0.25)
for i in range(X.shape[0]):
    if not P.contains(shapely.geometry.Point(X[i])):
        xi_cvx = np.dot(Z.T, A[i])
        line = np.vstack((X[i], xi_cvx))
        ax.plot(line[:,0], line[:,1], 'k-', alpha=0.5)
ax.scatter(X[:,0],X[:,1],marker='o',alpha=.99,color='darkgrey',edgecolors='k',
           zorder=2, label=r'Data Points $\mathbf{x}_i$')
ax.plot(Z[:,0],Z[:,1],'s',ms=8,color=col, label=r'Archetypes $\mathbf{Z}$')
ax.plot([], [], 'k-', alpha=0.5, label='Error')
ax.legend(loc='center right', bbox_to_anchor=(2, .5), borderaxespad=0,
          prop={'size': 20})
fig.tight_layout()


fig.savefig('/tmp/AA.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/AA.png', dpi=300, transparent=False, bbox_inches='tight')

################################################################################
# AA with Z trajectory
# This is Figure 1 of the paper.

np.random.seed(23)
X1 = np.random.multivariate_normal([-3,-1], np.eye(2), 50)
X2 = np.random.multivariate_normal([+3,-1], np.eye(2), 50)
X3 = np.random.multivariate_normal([+0,+6], np.eye(2), 50)
X4 = np.random.multivariate_normal([+0,+1], np.eye(2), 50)
X5 = np.random.multivariate_normal([+0,+3], np.eye(2), 50)

X = np.vstack((X1,X2,X3,X4,X5))


hullX = ConvexHull(X)
k = 4
col = 'royalblue'

np.random.seed(4)
ind = AApp.AApp(X, k)
Z_init = X[ind].copy()

fig, ax = plt.subplots(figsize=(6,4))
ax.set_aspect('equal')
ax.scatter(X[:,0],X[:,1],marker='o',alpha=.99,color='darkgrey',edgecolors='k',
           zorder=2, label=r'Data Points $\mathbf{x}_i$')
ax.plot(Z_init[:,0],Z_init[:,1],'s',ms=8,color='red')


M = 1000.0
Z = Z_init.copy()
Z_hist = [Z]
epsilon = 1e-3
n = X.shape[0]
A = np.zeros((n, k))  # convex combination for each data point xi, i=1..n
B = np.zeros((k, n))  # convex combination for each archetype  zj, j=1..k
rss = [99999]
Q = np.vstack((X.T, M * np.ones(n)))
for iteration in range(100):
    A = AA.ArchetypalAnalysis_compute_A(X, Z, M)
    Z = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, X), rcond=None)[0]
    for j in range(k):
        b, rnorm = nnls(Q, np.hstack((Z[j], M)))
        B[j] = b.T
    Z = np.dot(B, X)
    Z_hist.append(Z)
    rss.append(AA.RSS_Z(X, A, Z))
    converged = np.abs(rss[-1] - rss[-2]) / np.abs(rss[-1]) < epsilon
    if converged:
        break
A = AA.ArchetypalAnalysis_compute_A(X, Z, M)


fig, ax = plt.subplots(figsize=(6,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.add_artist(matplotlib.patches.Polygon(X[hullX.vertices], color='black', alpha=0.05, label=r'Convex Hull of $\mathbf{X}$'))
hullZ = ConvexHull(Z)
P = shapely.geometry.Polygon(Z).convex_hull
ax.add_artist(matplotlib.patches.Polygon(Z[hullZ.vertices], color=col, alpha=0.1, label=r'Convex Hull of $\mathbf{Z}$') )
for simplex in hullZ.simplices:
    ax.plot(Z[simplex,0], Z[simplex,1], ls='-', c=col, alpha=0.25)
for i in range(X.shape[0]):
    if not P.contains(shapely.geometry.Point(X[i])):
        xi_cvx = np.dot(Z.T, A[i])
        line = np.vstack((X[i], xi_cvx))
        ax.plot(line[:,0], line[:,1], 'k-', alpha=0.5)
ax.scatter(X[:,0],X[:,1],marker='o',alpha=.99,color='darkgrey',edgecolors='k',
           zorder=2, label=r'Data Points $\mathbf{x}_i$')
for j in range(k):
    Z_traj = np.array(Z_hist)[:,j,:]
    ax.plot(Z_traj[:,0],Z_traj[:,1],ls='-',color='darkorange')
ax.plot(Z_init[:,0],Z_init[:,1],'s',ms=8,color='darkorange', label=r'Initial Archetypes $\mathbf{Z}_{\operatorname{init}}$')
ax.plot([], [], ls='-', color='darkorange', label=r'Learning Path of $\mathbf{Z}$')
ax.plot(Z[:,0],Z[:,1],'s',ms=8,color=col, label=r'Learned Archetypes $\mathbf{Z}$')
ax.plot([], [], 'k-', alpha=0.5, label='Error')
ax.legend(loc='center right', bbox_to_anchor=(2.2, .5), borderaxespad=0,
          prop={'size': 20})
fig.tight_layout()

fig.savefig('/tmp/AA_init.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/AA_init.png', dpi=300, transparent=False, bbox_inches='tight')

################################################################################
# Idea
# This is Figure 2 of the paper.

n_points = 50
np.random.seed(23)

X1 = np.random.multivariate_normal([-3,-1], np.eye(2), n_points)
X2 = np.random.multivariate_normal([+3,-1], np.eye(2), n_points)
X3 = np.random.multivariate_normal([+0,+6], np.eye(2), n_points)
X4 = np.random.multivariate_normal([+0,+1], np.eye(2), n_points)
X5 = np.random.multivariate_normal([+0,+3], np.eye(2), n_points)

X = np.vstack((X1,X2,X3,X4,X5))

hullX = ConvexHull(X)
# test plot
fig, ax = plt.subplots(figsize=(8,6))
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.scatter(X[:,0],X[:,1],marker='o',alpha=.99,color='darkgrey',edgecolors='k')
ax.add_artist(matplotlib.patches.Polygon(X[hullX.vertices], color='black', alpha=0.05, label='convex hull'))


def add_stuff(ax, X, Z):
    n = X.shape[0]
    A = AA.ArchetypalAnalysis_compute_A(X, Z)
    MSE = AA.RSS_Z(X, A, Z) / n
    col = 'royalblue'
    ax.set_xlabel(f'MSE={MSE:2.2f}')
    if Z.shape[0] >= 3:
        hullZ = ConvexHull(Z)
        P = shapely.geometry.Polygon(Z).convex_hull
        ax.add_artist(matplotlib.patches.Polygon(Z[hullZ.vertices], color=col, alpha=0.1, label='convex hull') )
        for simplex in hullZ.simplices:
            ax.plot(Z[simplex,0], Z[simplex,1], ls='-', c=col, alpha=0.25)
        for i in range(X.shape[0]):
            if not P.contains(shapely.geometry.Point(X[i])):
                xi_cvx = np.dot(Z.T, A[i])
                line = np.vstack((X[i], xi_cvx))
                ax.plot(line[:,0], line[:,1], 'k-', alpha=0.3)
    else:
        for i in range(X.shape[0]):
            xi_cvx = np.dot(Z.T, A[i])
            line = np.vstack((X[i], xi_cvx))
            ax.plot(line[:,0], line[:,1], 'k-', alpha=0.15)
    ax.plot(Z[:,0],Z[:,1],'s',ms=8,color=col)


hullX = ConvexHull(X)

k = 4
seed = 19

np.random.seed(seed)
ind = baselines.Uniform(X, k)
Z_U = X[ind].copy()

np.random.seed(seed)
ind = baselines.FurthestSum(X, k)
Z_FS = X[ind].copy()

np.random.seed(seed)
ind = AApp.AApp(X, k)
Z_AApp = X[ind].copy()


fig, axv = plt.subplots(3, k, figsize=(8,6))
for ax in axv.flatten():
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(X[:,0],X[:,1],marker='o',s=14,alpha=.99,color='darkgrey',edgecolors='k')
    ax.add_artist(matplotlib.patches.Polygon(X[hullX.vertices], color='black', alpha=0.05, label='convex hull'))
axv[0,0].set_ylabel('Uniform')
axv[1,0].set_ylabel('FurthestSum')
axv[2,0].set_ylabel('AA++')
for i in range(k):
    axv[0,i].set_title(rf'$k$={i+1}')
    add_stuff(axv[0,i], X, Z_U[:i+1])
    add_stuff(axv[1,i], X, Z_FS[:i+1])
    add_stuff(axv[2,i], X, Z_AApp[:i+1])
fig.tight_layout()


fig.savefig('/tmp/idea.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/idea.png', dpi=300, transparent=False, bbox_inches='tight')


################################################################################
# k-means approx
# This is Figure 3 of the paper.

Z = np.array([[0.3,0.7],
              [1.1,1.1],
              [3.5,1.4],
              [3.8,1.2],
              [1.2,-2],
              [4.6,-5]])

X = np.array([[2.2,1.6],
              [3.3,0.8]])

A = AA.ArchetypalAnalysis_compute_A(X, Z)
hullZ = ConvexHull(Z)

col = 'royalblue'

fig, ax = plt.subplots(figsize=(6,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.3,4)
ax.set_ylim(0,2.4)
ax.set_aspect('equal')
ax.add_artist(matplotlib.patches.Polygon(Z[hullZ.vertices], color=col, alpha=0.1, label=r'Convex Hull of $\mathbf{Z}$') )
X_ = A@Z
closest_point = np.array([np.sum((Z-x)**2, axis=1) for x in X]).argmin(1)
ax.plot([X[0,0],X_[0,0]], [X[0,1],X_[0,1]], 'k-', alpha=0.75,
        label=r'True Distance')
ax.plot([X[0,0],Z[closest_point[0],0]], [X[0,1],Z[closest_point[0],1]],
        'k--', alpha=0.75)
ax.plot([X[1,0],Z[closest_point[1],0]], [X[1,1],Z[closest_point[1],1]],
        'k--', alpha=0.75, label=r'Approximated Distance')
ax.plot(Z[:,0],Z[:,1],'s',ms=8, c=col, label=r'Initialized Archetypes $\mathbf{Z}$')
ax.plot(X[0,0],X[0,1],'o',color='darkgreen')
ax.plot(X[1,0],X[1,1],'o',color='firebrick')
ax.legend(loc='upper center', ncol=2)


fig.savefig('/tmp/approx.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/approx.png', dpi=300, transparent=False, bbox_inches='tight')




################################################################################
# pre-processing
# This is Figure 9 of the paper.

X = np.array([[3.5,0.5],
              [-0.2,1.2],
              [0.1,-0.75],
              [-1.9,0.2],
              [-0.35,0.4],
              [1,-0.1],
              [0.1,0.2],
              [2,0.5],
              [1.5,-0.1],
              [0.2,1],
              [1,.5]])
X += 0.5

X_pre1 = X - X.mean(0)
X_pre1 = X_pre1 / X_pre1.max()

X_pre2 = StandardScaler().fit_transform(X)


tmp = np.vstack((X,X_pre1,X_pre2))
x0 = tmp.min(0)[0]*1.2
x1 = tmp.max(0)[0]*1.2
y0 = tmp.min(0)[1]*1.2
y1 = tmp.max(0)[1]*1.2


fig, axv = plt.subplots(1,3,figsize=(12,6))
for ax in axv:
    ax.set_aspect('equal')
    ax.set_xlim(x0,x1)
    ax.set_ylim(y0,y1)
    ax.grid()
for ax, D in zip(axv, (X_pre1, X, X_pre2)):
    hullD = ConvexHull(D)
    ax.add_artist(matplotlib.patches.Polygon(D[hullD.vertices], color='black', alpha=0.05, label=r'Convex Hull of $\mathbf{X}$'))
    ax.scatter(D[:,0],D[:,1],marker='o',alpha=.99,color='darkgrey',edgecolors='k',label=r'Data Set $\mathbf{X}$',zorder=2)
axv[1].legend(loc='lower center')

fig.savefig('/tmp/preprocessing.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/preprocessing.png', dpi=300, transparent=False, bbox_inches='tight')

