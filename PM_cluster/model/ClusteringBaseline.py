import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression 


class PopulationRegressor:
    def __init__(self):
        pass
    
    def fit(self, X, y, C):
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.reg = LinearRegression().fit(X, y)
        self.beta_hat = self.predict_beta(X)
        return self.beta_hat, np.ones_like(self.beta_hat)
        
    def predict_beta(self, X):
        beta_hat = self.reg.coef_
        beta_hats = np.tile(beta_hat, (len(X), 1))
        return beta_hats

    def predict_y(self, X):
        return self.reg.predict(X)


class ClusterRegressor:
    def __init__(self, K):
        self.K = K
        self.kmeans = KMeans(n_clusters=K)
        self.models = {k: PopulationRegressor() for k in range(K)}

    def setK(self, K):
        self.K = K
        self.kmeans = KMeans(n_clusters=K)
        self.models = {k: PopulationRegressor() for k in range(K)}
    
    def fit(self, X, y, C):
        self.kmeans.fit(C)
        labels = self.predict_l(C)
        for k in range(self.K):
            k_idx = labels == k
            X_k, y_k, C_k = X[k_idx], y[k_idx], C[k_idx]
            self.models[k].fit(X_k, y_k, C_k)
        self.beta_hat = self.predict_beta(X, C)
        return self.beta_hat, np.ones_like(self.beta_hat)
            
    def predict_l(self, C):
        return self.kmeans.predict(C)
    
    def predict_beta(self, X, C):
        labels = self.predict_l(C)
        beta_hat = np.zeros_like(X)
        for label in np.unique(labels):
            l_idx = labels == label
            X_l = X[l_idx]
            beta_hat[l_idx] = self.models[label].predict_beta(X_l)
        return beta_hat
    
    def predict_y(self, X, C):
        labels = self.predict_l(C)
        y_hat = np.zeros((len(X), 1))
        for label in np.unique(labels):
            l_idx = labels == label
            X_l = X[l_idx]
            y_hat[l_idx] = self.models[label].predict_y(X_l)
        return y_hat


if __name__ == '__main__':
    mse = lambda beta, beta_hat: ((beta - beta_hat)**2).mean()
    n = 1000
    c_dim = 10
    x_dim = 200
    y_dim = 1
    c = np.linspace(-1, 1, n)
    beta_1 = c.copy()
    beta_2 = np.concatenate((beta_1[:len(beta_1)//2]**2 - 1, 1 - beta_1[len(beta_1)//2:]**2), axis=0)
    beta = np.array([beta_1, beta_2]).T
    c = c[:, np.newaxis]
    idx = np.random.choice(np.arange(n), n)
    beta = beta[idx]
    c = c[idx]
    x = np.random.uniform(-1, 1, (n, 2))
    epsilon = np.random.normal(0, 1e-3, (n, 1))
    y = np.sum(beta * x, axis=1)[:, np.newaxis] + epsilon
    c_train, x_train, y_train, beta_train = c[:-10], x[:-10], y[:-10], beta[:-10]
    c_test, x_test, y_test, beta_test = c[-10:], x[-10:], y[-10:], beta[-10:]
    pop = PopulationRegressor()
    clr = ClusterRegressor(500)
    B_train, P_train = pop.fit(x_train, y_train, c_train)
    B_test = pop.predict_beta(x_test)
    print(f"Population Regressor Train MSE: {mse(beta_train, B_train)}")
    print(f"Population Regressor Test MSE:  {mse(beta_test, B_test)}")
    B, P = clr.fit(x_train, y_train, c_train)
    B_train, P_train = clr.fit(x_train, y_train, c_train)
    B_test = clr.predict_beta(x_test, c_test)
    print(f"Cluster Regressor Train MSE: {mse(beta_train, B_train)}")
    print(f"Cluster Regressor Test MSE: {mse(beta_test, B_test)}")
