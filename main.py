import collections

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from config import getClusteringResultsPath, getNonTimeSeriesDatasetsPath, getmKMeansName, getDbscanName
from utils.datasets_fetcher import getDatasetsFromFolder
from utils.persistor import resetStorage, storeResult, storeAlgoConfig
import numpy as np

# shows clustering info
def clusteringInfo(labels):
    if -1 in labels:
        print("including noise")
    counter = collections.Counter(labels)
    for l in set(labels):
        print(" - cluster", l, ' has ', counter.get(l), ' elements')

# obtain the data sets from the csv files
non_time_series_datasets = getDatasetsFromFolder(getNonTimeSeriesDatasetsPath())

# clustering algorithms
kmeans =  KMeans(n_clusters=3)
dbscan = DBSCAN(eps=0.2, min_samples=5)
algorithms = [(getmKMeansName(), kmeans),
              (getDbscanName(), dbscan)]

# iterate over the data sets
for datIndx in range(len(non_time_series_datasets)):  # row index
    X = non_time_series_datasets[datIndx]['dataset']
    dName = non_time_series_datasets[datIndx]['name']
    k = non_time_series_datasets[datIndx]['k']
    baseFolder = getClusteringResultsPath() + dName + '/'
    resetStorage(baseFolder)
    X = StandardScaler().fit_transform(X)
    # iterate over the algorithms
    for algIndx in range(len(algorithms)):  # column index
        # normalize dataset for easier parameter selection
        algo = algorithms[algIndx][1]
        algoName = algorithms[algIndx][0]
        # if the algorithm is 'kmeans', k must be set specifically for the data set
        if algoName.lower() == 'kmeans':
            algo.set_params(n_clusters=k)
        # assign labels
        labels = algo.fit_predict(X)
        print("  " ,len(set(labels)), " clusters")
        clusteringInfo(labels)
        # result contains every scaled element with the corresponding label
        result = np.c_[X,labels]
        # store result
        folder = baseFolder + algoName + '/'
        storeResult(result, folder)
        # store algo config
        paramsDict = algo.get_params()
        storeAlgoConfig(paramsDict, folder)
        print("\n")

