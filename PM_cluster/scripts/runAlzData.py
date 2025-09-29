__author__ = 'Haohan Wang'

from model.personalizedModel import PersonalizedThroughMixedModel

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def run_exp():
    Y = np.load('../data/ADNI/prepared/labels.npy')
    X = np.load('../data/ADNI/prepared/snps.npy')
    C = np.load('../data/ADNI/prepared/embeddings.npy')

    covariate = np.load('../data/ADNI/prepared/covariates.npy')
    covariate = normalize(covariate, axis=0)

    lr = LinearRegression()
    lr.fit(covariate, Y)
    R = Y - lr.predict(covariate)

    print (R.shape)
    print (np.mean(R))
    R = (R - np.mean(R)) / np.std(R)

    pl = PersonalizedThroughMixedModel()

    B, P = pl.fit(X, R, C)

    np.save('pvalues.npy', P)

    plt.imshow(P)
    plt.show()

def globalFDR(pvalues, alpha=0.01):
    tmp = pvalues.flatten().tolist()
    tmp = sorted(tmp)
    threshold = 1e-8
    n = len(tmp)
    for i in range(n):
        if tmp[i] < (i+1)*alpha/n:
            threshold = tmp[i]
    pvalues[pvalues>threshold] = 1
    return pvalues

def results_study():

    markers = [line.strip() for line in open('../data/ADNI/SNPs/exonMarkers.txt')]
    Y = np.load('../data/ADNI/prepared/labels.npy')
    sids = [line.strip() for line in open('../data/ADNI/prepared/ids.txt')]

    pvalues = np.load('pvalues.npy')

    pvalues = globalFDR(pvalues)

    count = np.zeros_like(pvalues)
    count[pvalues<=0.01] = 1

    count_individual = np.sum(count, 1)
    for i in range(count_individual.shape[0]):
        print (Y[i], count_individual[i])

    discussed = {}

    f = open('results_individual.csv', 'w')

    for i in range(len(sids)):
        sid = sids[i].split('#')[0]
        if sid not in discussed:

            if count_individual[i] < 6000 and Y[i] > 0.5:
                discussed[sid] = 0

                pvalue = pvalues[i]
                indices = np.argsort(pvalue)

                f.writelines(sid + ',' + str(Y[i]) )
                for idx in indices:
                    if pvalue[idx] < 0.01:
                        f.writelines(',' + markers[idx] + ',' + str(pvalue[idx]))
                f.writelines('\n')

    f.close()

    pvalues_max = np.max(pvalues, 0)
    print (pvalues_max.shape)
    print (np.min(pvalues_max))

    f = open('results_aggregrated.csv', 'w')
    idx = np.argsort(pvalues_max)
    for i in idx:
        if pvalues_max[i] < 0.01:
            f.writelines(markers[i] + ',' + pvalues_max[i] + '\n')
    f.close()


    # print (idx)
    # print (len(idx))



if __name__ == '__main__':
    # run_exp()
    results_study()