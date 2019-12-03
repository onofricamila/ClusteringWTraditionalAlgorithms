import json

csvDatasetsPath = None
timeSeriesDatasetsPath = None
nonTimeSeriesDatasetsPath = None
algoNames = None
clusteringResultsPath = None

def fetchConfig():
    # we use the global key word to being able to change the values of the variables declared outside the function
    global csvDatasetsPath
    global timeSeriesDatasetsPath
    global nonTimeSeriesDatasetsPath
    global clusteringResultsPath
    global algoNames

    configFilePath = "/home/camila/Desktop/TESIS/DATA/config.json"
    with open(configFilePath) as f:
        data = json.load(f)
    # fill variables
    clusteringResultsPath = data.get("clusteringResultsPath")
    csvDatasetsPath = data.get("csvDatasetsPath")
    timeSeriesDatasetsPath = data.get("timeSeriesDatasetsPath")
    nonTimeSeriesDatasetsPath = data.get("nonTimeSeriesDatasetsPath")
    algoNames = data.get("algoNames")


def getCsvDatasetsPath():
    if csvDatasetsPath is not None:
        return csvDatasetsPath
    # else
    fetchConfig()
    return csvDatasetsPath


def getClusteringResultsPath():
    if clusteringResultsPath is not None:
        return clusteringResultsPath
    # else
    fetchConfig()
    return clusteringResultsPath


def getNonTimeSeriesDatasetsPath():
    if nonTimeSeriesDatasetsPath is not None:
        return nonTimeSeriesDatasetsPath
    # else
    fetchConfig()
    return nonTimeSeriesDatasetsPath


def getTimeSeriesDatasetsPath():
    if timeSeriesDatasetsPath is not None:
        return timeSeriesDatasetsPath
    # else
    fetchConfig()
    return timeSeriesDatasetsPath

def getmKMeansName():
    key = "kmeans"
    kmeansName = algoNames.get(key)
    if (kmeansName != None):
        return kmeansName
    # else
    fetchConfig()
    return algoNames.get(key)


def getDbscanName():
    key = "dbscan"
    dbscanName = algoNames.get(key)
    if (dbscanName != None):
        return dbscanName
    # else
    fetchConfig()
    return algoNames.get(key)
