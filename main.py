import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import collections
from DBCV.validity import validity_index
from sklearn.preprocessing import StandardScaler
from datasets import datasets
# set matplotlib backend to Qt5Agg to make figure window maximizer work
import matplotlib
matplotlib.use('Qt5Agg')

# shows clustering info
def clusteringInfo(labels):
    if -1 in labels:
        print("including noise")
    counter = collections.Counter(labels)
    for l in set(labels):
        print(" - cluster", l, ' has ', counter.get(l), ' elements')

# clustering algorithms
kmeans =  KMeans(n_clusters=3)
dbscan = DBSCAN(eps=0.2, min_samples=5)
algorithms = [("kmeans", kmeans),
              ("dbscan", dbscan)]

# configure fig
rows = len(datasets)
cols = len(algorithms)
fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

# iterate over the data sets
for r in range(rows):  #row index
    # iterate over the algorithms
    for c in range(cols):  # column index
        X, y = datasets[r]
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
        algo = algorithms[c][1]
        algoName = algorithms[c][0]
        labels = algo.fit_predict(X)
        ax = axes[r, c]
        ax.scatter(X[:, 0], X[:, 1], s=10, c=labels, cmap="nipy_spectral")
        print(f'plotting at index [ {r} , {c}]')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        # if it's the first clustering of an algorithm, print the algo name
        if r == 0:
            ax.set_title(algoName, size=18)
        # show clustering info
        print("  " ,len(set(labels)), " clusters")
        clusteringInfo(labels)
        # obtain scores
        try:
            score = validity_index(X=X, labels=labels, metric=euclidean, per_cluster_scores=True, )
        except ValueError as e:
            print(' Failed to calculate DBCV Index: ' + str(e))
            score = None
        print(" --> score: ", score)
        # add DBCV score to axes
        ax.text(x=.99, y=.80, s="DBCV " + str(round(score[0], 2)), fontsize=10, transform=ax.transAxes, horizontalalignment='right')
        print("\n")
    print("\n")


# show both subplots
fig.canvas.manager.window.showMaximized()
plt.show()

print("\nHigher score -> better clustering result (remember DBCV score is between -1 and 1)")
