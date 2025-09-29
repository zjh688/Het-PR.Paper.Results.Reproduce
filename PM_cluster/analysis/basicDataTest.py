__author__ = 'Haohan Wang'

import numpy as np

snps = np.load('../data/ADNI/prepared/snps.npy')

print (snps.shape)

cov = np.load('../data/ADNI/prepared/covariates.npy')

print (cov.shape)

labels = np.load('../data/ADNI/prepared/labels.npy')

print (labels.shape)

print (labels)