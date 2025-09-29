__author__ = 'Haohan Wang'

import numpy as np

def readInResults():
    text = [line.strip() for line in open('quick_results.txt')]
    results = []
    for line in text:
        if line.startswith('['):
            items = line[1:-1].split(', ')
            result = [float(t) for t in items]
            results.append(result)

    return results

def writeOutResults():
    results = readInResults()

    f = open('init_results.csv', 'w')
    f.writelines('n,p,k,blockNum,effectSizeScale,noiseScale,distanceNoiseScale,distanceReshufflePercentage,mode_regressionModel,seed,p_mse, p_auc, b_mse, b_auc, ars\n')
    for rl in results:
        for r in rl:
            f.writelines(str(r) + ',')
        f.writelines('\n')

    f.close()

if __name__ == '__main__':
    writeOutResults()