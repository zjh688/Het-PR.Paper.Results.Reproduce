__author__ = 'Haohan Wang'

import numpy as np

from dataGeneration.generationData import generatePersonalizedDataIID

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ars

def evaluation_partition(oracle_Z, predicted):
    k = int(np.max(oracle_Z)+1)

    km = KMeans(n_clusters=k)
    try:
        r = km.fit_predict(predicted)
    except:
        r = np.ones(predicted.shape[0])

    print (r)
    print (Z)

    return ars(oracle_Z, r)

if __name__ == '__main__':

    n = 100
    p = 50
    k = 5
    blockNum = 10
    effectSizeScale = 5
    noiseScale = 0.1
    distanceNoiseScale = 0
    distanceReshufflePercentage = 0
    mode_regressionModel = 'lmm'
    useOracleDistance = True
    seed = 1

    X, B, y, C, Z = generatePersonalizedDataIID(n=n, p=p, k=k, blockNum=blockNum, effectSizeScale=effectSizeScale,
                                                noiseScale=noiseScale, distanceNoiseScale=distanceNoiseScale,
                                                distanceReshufflePercentage=distanceReshufflePercentage)

    B1 = np.random.random(B.shape)

    print (evaluation_partition(Z, B1))