__author__ = 'Haohan Wang'

import numpy as np
from numpy import linalg


class PAN:
    def __init__(self, lam=1., lr=1., tol=1e-5):
        self.lam = lam
        self.lr = lr
        self.tol = tol
        self.decay = 0.5
        self.maxIter = 500

    def setLambda(self, lam):
        self.lam = lam

    def setLearningRate(self, lr):
        self.lr = lr

    def setMaxIter(self, a):
        self.maxIter = a

    def setTol(self, t):
        self.tol = t

    def fit(self, X, y, C=None):
        self.beta = np.zeros_like(X)
        self.p = np.ones_like(X)
        for i in range(X.shape[0]):
            resi_max = np.inf
            for lmbd in np.logspace(-3, 3, 7):
                self.setLambda(lmbd)
                r, b = self.fit_oneStep(X, y, X[i, :])
                if r < resi_max:
                    resi_max = r
                    self.beta[i, :] = b.reshape(X.shape[1])
        return self.beta, self.p

    def fit_oneStep(self, X, y, x):
        lr_reset = self.lr
        shp = X.shape
        beta = np.random.random([shp[1], 1]) * 1e-10
        resi_prev = np.inf
        resi = self.cost(X, y, x, beta)

        step = 0
        while np.abs(resi_prev - resi) > self.tol and step < self.maxIter:
            keepRunning = True
            resi_prev = resi
            runningStep = 0
            while keepRunning and runningStep < 10:
                runningStep += 1
                prev_beta = beta
                grad = self.gradient(X, y, x, beta)
                beta = beta - grad * self.lr
                keepRunning = self.stopCheck(prev_beta, beta, grad, X, y)
                if keepRunning:
                    self.lr = self.decay * self.lr
            step += 1
            resi = self.cost(X, y, x, beta)
        self.lr = lr_reset
        return resi, beta

    def cost(self, X, y, x, beta):
        return 0.5 * np.sum(np.square(y - (np.dot(X, beta)).transpose())) + \
               self.lam / np.dot(x, x.T) * np.dot(np.dot(beta.T, np.dot(x.T, x)), beta) / np.dot(beta.T, beta)

    def gradient(self, X, y, x, beta):
        y = y[:, np.newaxis]
        x = x[:, np.newaxis]
        A = x @ x.T
        beta_norm = beta.T @ beta
        grad = -2 * y.T @ X + 2 * beta.T @ X.T @ X + \
                2 * self.lam / (x.T @ x) * (1 / beta_norm * beta.T @ A - (beta.T @ A @ beta / beta_norm ** 2) * beta.T)
        return grad.T

        # print (np.dot(x.T, x))
        # print (np.dot(x.T, x).shape)

        # return -np.dot(X.transpose(), (y.reshape((y.shape[0], 1)) - (np.dot(X, beta)))) + \
        #        2 * self.lam / np.dot(x, x.T) * (
        #                np.dot(x.T, x) / np.dot(beta.T, beta) - np.dot(np.dot(beta.T, np.dot(x.T, x)), beta) / (
        #            np.dot(beta.T, beta)) ** 2) * beta

    def predict(self, X):
        return np.dot(X, self.beta)

    def getBeta(self):
        return self.beta

    def stopCheck(self, prev, new, pg, X, y):
        if np.square(linalg.norm((y - (np.dot(X, new))))) <= \
                np.square(linalg.norm((y - (np.dot(X, prev))))) + np.dot(pg.transpose(), (
                new - prev)) + 0.5 * self.lam * np.square(linalg.norm(prev - new)):
            return False
        else:
            return True

if __name__ == '__main__':
    X = np.random.random([10, 5])
    y = np.random.random([10])

    print ('get started')
    pan = PAN()
    beta, p = pan.fit(X, y, None)
    print (beta)
