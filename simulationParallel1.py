# -*- coding: utf-8 -*-
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
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings

# ======================= 全局忽略 FutureWarning =======================
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= 定义结果存放目录（绝对路径） =======================
OUTPUT_DIR = Path.home() / "PM_cluster" / "simulation1" / "simpleDataStructure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= 原函数：不改内容 =======================

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
            log_writer.writelines('\t' + str(r))

    saveName = getSaveName(n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale, distanceReshufflePercentage,
                           mode_regressionModel,
                           useOracleDistance, seed)

    np.save(OUTPUT_DIR / (saveName + 'X.npy'), X)
    np.save(OUTPUT_DIR / (saveName + 'B.npy'), B)
    np.save(OUTPUT_DIR / (saveName + 'y.npy'), y)
    np.save(OUTPUT_DIR / (saveName + 'C.npy'), C)
    np.save(OUTPUT_DIR / (saveName + 'Z.npy'), Z)
    np.save(OUTPUT_DIR / (saveName + 'estimated.npy'), estimatedMatrix)

# ======================= 新增：轻量日志缓冲器 =======================

class LogBuffer:
    def __init__(self):
        self._buf = []
    def writelines(self, s):
        self._buf.append(s)
    def getvalue(self):
        return ''.join(self._buf)

# ======================= 并行封装 + 断点续跑 =======================

def _task_runner(params):
    (n, p, k, blockNum, effectSizeScale, noiseScale, distanceNoiseScale,
     distanceReshufflePercentage, mode_regressionModel, seed) = params

    header = (str(n) + '\t' + str(p) + '\t' + str(k) + '\t' + str(blockNum) + '\t' +
              str(effectSizeScale) + '\t' + str(noiseScale) + '\t' + str(distanceNoiseScale) + '\t' +
              str(distanceReshufflePercentage) + '\t' + str(seed))

    # === 检查结果文件是否都存在，存在就跳过 ===
    saveName = getSaveName(n, p, k, blockNum, effectSizeScale, noiseScale,
                           distanceNoiseScale, distanceReshufflePercentage,
                           mode_regressionModel, True, seed)
    expected_files = [OUTPUT_DIR / (saveName + suffix) 
                      for suffix in ["X.npy","B.npy","y.npy","C.npy","Z.npy","estimated.npy"]]
    if all(f.exists() for f in expected_files):
        return header + "\tSKIPPED\n"

    # === 否则照常运行 ===
    buf = LogBuffer()
    simulation(n, p, k, blockNum, effectSizeScale, noiseScale,
               distanceNoiseScale, distanceReshufflePercentage,
               mode_regressionModel, True, seed, buf)
    return header + buf.getvalue() + '\n'

# ======================= 主程序 =======================

if __name__ == '__main__':
    results = []

    param_list = []
    for n in [1000]:
        for p in [50, 100, 500]:
            if p > n:
                continue
            for d in [float(1/p), 0.05, 0.1]:
                k = int(d*p)
                for blockNum in [5, 10, 50]:
                    for effectSizeScale in [1]:
                        for noiseScale in [0.1]:
                            for distanceNoiseScale in [0]:
                                for distanceReshufflePercentage in [0]:
                                    for mode_regressionModel in ['lmm']:
                                        for seed in [1, 2, 3]:
                                            param_list.append((
                                                n, p, k, blockNum, effectSizeScale, noiseScale,
                                                distanceNoiseScale, distanceReshufflePercentage,
                                                mode_regressionModel, seed
                                            ))

    max_workers = int(os.environ.get('FOS_SIM_NWORKERS', os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        lines = list(ex.map(_task_runner, param_list))

    with open('logs1.tsv', 'w') as log_writer:
        for line in lines:
            log_writer.write(line)
