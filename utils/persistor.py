import json

import numpy as np
import os
from config import getClusteringResultsPath, getTimeSeriesDatasetsPath
import shutil


def resetStorage(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def createDirectoryIfNotExists(folder):
    # check if resourcesFolder needs to be created
    if not os.path.exists(folder):
        os.makedirs(folder)


def storeResult(result, folder):
    createDirectoryIfNotExists(folder)
    targetFile = folder + 'result' + '.csv'
    np.savetxt(targetFile, result, delimiter=',',)


def storeAlgoConfig(dict, folder):
    createDirectoryIfNotExists(folder)
    file = folder + 'algoConfig.json'
    with open(file, 'w') as outfile:
        json.dump(dict, outfile)

