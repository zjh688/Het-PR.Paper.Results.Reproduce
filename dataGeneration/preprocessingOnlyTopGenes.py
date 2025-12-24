__author__ = 'Haohan Wang'

import numpy as np

def selectingOnlyTopGenes(X, top=1000):

    genes = [line.strip() for line in open('../data/Harvard/toStudy/gids.txt')]

    topGenes = [line.strip().split('\t')[2] for line in open('../data/geneCardInformation.txt')][1:]

    idx = []
    for gene in topGenes:
        try:
            i = genes.index(gene)
            idx.append(i)
        except:
            continue
        if len(idx) == top:
            break

    print (idx)
    print (len(idx))

    X_ = np.zeros((X.shape[0], top))

    f = open('../data/Harvard/topGenes/topGenes.txt', 'w')

    for i in range(top):
        X_[:, i] = X[:, idx[i]]
        f.writelines(genes[idx[i]] + '\n')

    f.close()

    return X_



def regressingGenes(regionName):

    from sklearn.linear_model import LinearRegression

    age = np.load('../data/Harvard/toStudy/'+ regionName +'_age.npy')
    gender = np.load('../data/Harvard/toStudy/'+ regionName +'_gender.npy')
    ph = np.load('../data/Harvard/toStudy/' + regionName + '_ph.npy')
    pmi = np.load('../data/Harvard/toStudy/' + regionName + '_pmi.npy')
    rin = np.load('../data/Harvard/toStudy/' + regionName + '_rin.npy')

    X = np.load('../data/Harvard/toStudy/' + regionName + '_gene.npy')

    age = age.reshape([age.shape[0], 1])
    gender = gender.reshape([gender.shape[0], 1])
    ph = ph.reshape([ph.shape[0], 1])
    pmi = pmi.reshape([pmi.shape[0], 1])
    rin = rin.reshape([rin.shape[0], 1])

    c = np.append(age, gender, 1)
    c = np.append(c, ph, 1)
    c = np.append(c, pmi, 1)
    c = np.append(c, rin, 1)

    X = selectingOnlyTopGenes(X)

    X_new = np.zeros_like(X)

    lr = LinearRegression()

    for i in range(X.shape[1]):
        lr.fit(c, X[:, i])
        X_new[:, i] = X[:, i] - lr.predict(c)

    np.save('../data/Harvard/topGenes/'+ regionName +'_Gene_residues.npy', X_new)

def preparingTopGenes(regionName):
    X = np.load('../data/Harvard/toStudy/' + regionName + '_gene.npy')

    X = selectingOnlyTopGenes(X)

    np.save('../data/Harvard/topGenes/'+ regionName +'_genes.npy', X)

if __name__ == '__main__':

    for rn in ['CR', 'PFC', 'VC']:
        # regressingGenes(rn)
        preparingTopGenes(rn)

    # selectingOnlyTopGenes(None)