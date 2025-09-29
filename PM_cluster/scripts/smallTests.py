__author__ = 'Haohan Wang'

from dataGeneration.generationData import generatePersonalizedDataIID
from model.personalizedModel import PersonalizedThroughMixedModel
from model.DistanceMatchingBaseline import DistanceMatchingRegressor
from model.PersonalizedAngularRegression import PAN
from model.ClusteringBaseline import ClusterRegressor, PopulationRegressor

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import adjusted_rand_score as ars

from sklearn.cluster import KMeans

import inspect


def evaluation_beta(oracle_beta, predicted_beta, flag='mse'):
    if flag == 'mse':
        return mse(oracle_beta, predicted_beta)

    elif flag == 'auc':
        I = np.zeros_like(oracle_beta)
        I[oracle_beta != 0] = 1
        I = I.ravel()
        predicted_beta = predicted_beta.ravel()
        return auroc(I, np.abs(predicted_beta))


def evaluation_pvalues(oracle_beta, pvalues, flag='mse'):
    I = np.zeros_like(oracle_beta)
    I[oracle_beta != 0] = 1
    I_1 = np.zeros_like(pvalues)
    I_1[pvalues < 0.05] = 1

    if flag == 'mse':
        return mse(I, I_1)
    elif flag == 'auc':
        I = I.ravel()
        pvalues = pvalues.ravel()
        return auroc(I, -np.log(pvalues))


def evaluation_partition(oracle_Z, predicted):
    k = int(np.max(oracle_Z)+1)

    km = KMeans(n_clusters=k)
    try:
        r = km.fit_predict(predicted)
    except:
        r = np.ones(predicted.shape[0])

    return ars(oracle_Z, r)

def getSaveName(n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale, distanceReshufflePercentage,
                mode_regressionModel,
                useOracleDistance, seed):

    name = ''
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        name = name + str(values[i]) + '_'
    return name


def simulation(n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale, distanceReshufflePercentage,
               mode_regressionModel,
               useOracleDistance, seed, log_writer):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        print(" %s = %s" % (i, values[i]))

    np.random.seed(seed)

    X, B, y, C, Z = generatePersonalizedDataIID(n=n, p=p, k=k, blockNum=blockNum, effectSizeScale=effectSizeScale,
                                                noiseScale=noiseScale, distanceNoiseScale=distanceNoiseScale,
                                                distanceReshufflePercentage=distanceReshufflePercentage)

    pl_pvalue = PersonalizedThroughMixedModel(mode_regressionModel='lmm')
    pl_beta = PersonalizedThroughMixedModel(mode_regressionModel='lr')
    dmr = DistanceMatchingRegressor()
    par = PAN(lam=1.0)
    cr = ClusterRegressor(3)
    pr = PopulationRegressor()

    modelsList = [pl_pvalue, pl_beta, dmr, par, cr, pr]

    estimatedMatrix = np.zeros((len(modelsList),) + B.shape)

    for i in range(len(modelsList)):
        model = modelsList[i]

        if i == 4:
            model.setK(int(np.max(Z)+1))

        if useOracleDistance:
            B_1, P = model.fit(X, y, C)
        else:
            B_1, P = model.fit(X, y, np.random.random(size=C.shape))

        if i == 0:
            estimatedMatrix[i] = P
        else:
            estimatedMatrix[i] = B_1

        if (P == 0).all():
            p_mse = np.nan
            p_auc = np.nan
        else:
            p_mse = evaluation_pvalues(B, P, flag='mse')
            p_auc = evaluation_pvalues(B, P, flag='auc')
        b_mse = evaluation_beta(B, B_1, flag='mse')/k
        b_auc = evaluation_beta(B, B_1, flag='auc')

        ars1 = evaluation_partition(Z, P)
        ars2 = evaluation_partition(Z, B)

        rl = (p_mse, p_auc, b_mse, b_auc, ars1, ars2)

        for r in rl:
            log_writer.writelines( '\t' + str(r))

    saveName = getSaveName(n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale, distanceReshufflePercentage,
                           mode_regressionModel,
                           useOracleDistance, seed)

    np.save('../simulation/simpleDataStructure/' + saveName + 'X.npy', X)
    np.save('../simulation/simpleDataStructure/' + saveName + 'B.npy', B)
    np.save('../simulation/simpleDataStructure/' + saveName + 'y.npy', y)
    np.save('../simulation/simpleDataStructure/' + saveName + 'C.npy', C)
    np.save('../simulation/simpleDataStructure/' + saveName + 'Z.npy', Z)
    np.save('../simulation/simpleDataStructure/' + saveName + 'estimated.npy', estimatedMatrix)



if __name__ == '__main__':
    # n = 100
    # p = 50  # since now the regressor is set up to be a linear regression, let's only worry about the cases when p < n
    # k = 5
    # blockNum = 10
    # effectSizeScale = 5
    # noiseScale = 0.1
    # distanceNoiseScale = 0
    # distanceReshufflePercentage = 0
    # mode_regressionModel = 'lmm'
    # useOracleDistance = True
    # seed = 1

    results = []

    log_writer = open('logs.tsv', 'w')

    for n in [100]:
        for p in [25]:
            if p > n:
                continue
            for d in [0.05]:
                k = int(d*p)
                for blockNum in [5]:
                    for effectSizeScale in [1]:
                        for noiseScale in [0.1]:
                            for distanceNoiseScale in [0]:
                                for distanceReshufflePercentage in [0]:
                                    for mode_regressionModel in ['lmm']:
                                        for seed in [1]:

                                            log_writer.writelines(str(n) + '\t'
                                                             + str(p) + '\t'
                                                             + str(k) + '\t'
                                                             + str(blockNum) + '\t'
                                                             + str(effectSizeScale) + '\t'
                                                             + str(noiseScale) + '\t'
                                                             + str(distanceNoiseScale) + '\t'
                                                             + str(distanceReshufflePercentage) + '\t'
                                                             + str(seed))

                                            result = simulation(n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale, distanceReshufflePercentage,
                                                               mode_regressionModel,
                                                               True, seed, log_writer)

                                            log_writer.writelines('\n')

                                            results.append(result)
                                        log_writer.flush()

    log_writer.close()
    # results = np.array(results)
    #
    # np.save('results.npy', results)
