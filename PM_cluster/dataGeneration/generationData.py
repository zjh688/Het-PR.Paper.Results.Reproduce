__author__ = 'Haohan Wang'

import numpy as np


def generatePersonalizedDataIID(n=200, p=1000, k=10, blockNum=None, effectSizeScale=5, noiseScale=0.1,
                                distanceNoiseScale=0, distanceReshufflePercentage=0.0):
    '''
    :param n: number of subjects (samples)
    :param p: number of features (covariates)
    :param blockNum: the underlying population (in terms of coefficients): how many different clusters of betas
    :param effectSizeScale: scalar of effectSize (beta)
    :param noiseScale: scalar of noises
    :param distanceNoiseScale: scalar of the different scale of the measurement of the noises,
                to simulate the fact the the real-world observation of the distances will be different from the true distance,
                the scalar controls how different it could be
    :return:
    '''

    X = np.random.random([n, p])

    if blockNum is None:
        blockNum = n

    B0 = np.zeros(shape=[blockNum, p])

    E0 = np.random.normal(size=[blockNum, k])
    E0 = E0 * effectSizeScale
    E0[E0 > 0] += effectSizeScale
    E0[E0 < 0] -= effectSizeScale

    for i in range(blockNum):
        idx = np.random.choice(p, size=k, replace=False)
        B0[i, idx] = E0[i]

    if blockNum != n:
        B = np.repeat(B0, n / blockNum, axis=0)
    else:
        B = B0

    ## oracle partition
    Z = []
    for i in range(blockNum):
        Z.extend([i for j in range(int(n / blockNum))])
    Z = np.array(Z)

    # B.sort(1)
    # B.sort(0)
    tmp = X * B

    y = np.sum(tmp, 1)

    y = y + np.random.normal(size=[n]) * noiseScale

    C = np.copy(B)

    if distanceNoiseScale != 0:
        C += np.random.normal(size=[n, p]) * distanceNoiseScale
    if distanceReshufflePercentage != 0:
        idx1 = np.random.choice(n, size=int(n*distanceReshufflePercentage), replace=False)
        idx2 = np.random.choice(n, size=int(n*distanceReshufflePercentage), replace=False)

        C[idx1, :], C[idx2, :] = C[idx2, :], C[idx1, :]

    return X, B, y, C, Z


if __name__ == '__main__':
    X, B, y, C, Z = generatePersonalizedDataIID(n=1000, p=500, blockNum=500, distanceReshufflePercentage=0.5)

    print(np.sum(B, 1))

    print(Z)

    # from matplotlib import pyplot as plt
    #
    # plt.imshow(B)
    # plt.show()
