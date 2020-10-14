# Gaussian process

# 1. get K: take x_train and compute kernel of x_train with itself
# 2. get the cholesky decomposition of K to get L*L**T


from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from  sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
from scipy import stats

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import sklearn
from sklearn.kernel_approximation import Nystroem

import numpy as np

import GPy

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""



def log_like(r, K):
    """
    The multivariate Gaussian ln-likelihood (up to a constant) for the
    vector ``r`` given a covariance matrix ``K``.

    :param r: ``(N,)``   The residual vector with ``N`` points.
    :param K: ``(N, N)`` The square (``N x N``) covariance matrix.

    :returns lnlike: ``float`` The Gaussian ln-likelihood.

    """
    # Erase the following line and implement the Gaussian process
    # ln-likelihood here.
    return -0.5 * (np.dot(r, np.linalg.solve(K, r)) + np.linalg.slogdet(K)[1])



class Model():

    def __init__(self):
        self.train_x=0
        self.train_y=0

        self.s = 0.00005  # noise variance.
        self.L = 0
        pass

    def kernel_exponential_cov(self, a, b):
        """ GP squared exponential kernel """
        kernelParameter = 10000
        param2 = 1
        # sum is added in case we have a square a or
        # (a+b)^2=a^2+b^2-2ab
        sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return param2 * np.exp(-.5 * (1 / kernelParameter) * sqdist)

    # def kernel_exponential_cov_2(x, y, params):
    #     return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y) ** 2)

    def nystrom(self, X_train, X_test, gamma, c=500, k=200, seed=44):
        rng = np.random.RandomState(seed)
        n_samples = X_train.shape[0]
        idx = rng.choice(n_samples, c)

        X_train_idx = X_train[idx, :]
        W = rbf_kernel(X_train_idx, X_train_idx, gamma=gamma)

        u, s, vt = linalg.svd(W, full_matrices=False)
        u = u[:, :k]
        s = s[:k]
        vt = vt[:k, :]

        M = np.dot(u, np.diag(1 / np.sqrt(s)))

        C_train = rbf_kernel(X_train, X_train_idx, gamma=gamma)
        C_test = rbf_kernel(X_test, X_train_idx, gamma=gamma)

        X_new_train = np.dot(C_train, M)
        X_new_test = np.dot(C_test, M)

        return X_new_train, X_new_test

    def predict(self, test_x):
        # 1
        # Lk = np.linalg.solve(self.L, self.kernel(self.train_x, test_x))  # L is (10,10) and kernel (10,50)
        # mu = np.dot(Lk.T, np.linalg.solve(self.L, self.train_y))

        # 2

        from numpy.linalg import inv

        #
        # params = [1, 10]
        # # sigma_1 = kernel_exponential_cov(x, x, params)
        # sigma_1 = kernel(x, x)
        #
        # def predict(x_j, data, kernel, params, sigma, t):
        #
        #
        #     # y is not a lable here , just the second term in the kernel(x,y)
        #     k = [kernel_exponential_cov(x_j, x_i, params) for x_i in data]
        #
        #     Sinv = np.linalg.inv(sigma)
        #     y_pred = np.dot(k, Sinv).dot(t)
        #     sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
        #
        #     return y_pred, sigma_new
        #
        #
        # predictions = [predict(i, x, kernel_exponential_cov, params, sigma_1, self.train_y) for i in test_x]
        #
        # 3


        X_train = self.train_x
        #
        # feature_map_nystroem = sklearn.kernel_approximation.Nystroem(gamma=.2,random_state = 1, n_components = 300)
        # data_transformed = feature_map_nystroem.fit_transform(X_train)
        #
        Y_train = self.train_y
        X_s = test_x

        l=1
        sigma_f=10
        #
        # K = self.kernel_exponential_cov_2(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
        # K_s = self.kernel_exponential_cov_2(X_train, X_s, l, sigma_f)
        # K_ss = self.kernel_exponential_cov_2(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
        # K_inv = inv(K)

        sigma_y=1
        gamma = 0.1

        X_train, X_s = self.nystrom(X_train, test_x, gamma, c=500, k=300, seed=44)

        K = self.kernel_exponential_cov(X_train, X_train) + sigma_y ** 2 * np.eye(len(X_train))
        K_s = self.kernel_exponential_cov(X_train, X_s)
        K_ss = self.kernel_exponential_cov(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = inv(K)

        # Equation (4)
        mu_s = K_s.T.dot(K_inv).dot(Y_train)

        # Equation (5)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        #
        # # Lk = np.linalg.solve(self.L, kernel(self.train_x, test_x))  # L is (10,10) and kernel (10,50)
        # # mu = np.dot(Lk.T, np.linalg.solve(self.L, self.train_y))

        #v4

        #X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
        #kernel = sklearn.gaussian_process.kernels.(length_scale=1.0) #+ WhiteKernel()
        kernel = sklearn.gaussian_process.kernels.RationalQuadratic() #+ WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state = 0).fit(self.train_x, self.train_y)
        #gpr.score(X_train, Y_train)
        test_y = gpr.predict(test_x)


        # v5

        # hyperparameters


        m = GPy.kern.RBF(input_dim=2, variance=2.5, lengthscale=0.15)
        y_train = self.train_y.reshape(-1,1)
        m = GPy.models.GPRegression(self.train_x, y_train, m)
        #m= GPy.models.SparseGPRegression(self.train_x, y_train, m)
        #m.plot()
        m.Gaussian_noise = 0.2
        m.optimize()
        mu, C = m.predict(test_x, full_cov=True)
        #samp = np.random.multivariate_normal(mu.reshape(-1,), C, 100).T
        posteriorTestY = m.posterior_samples_f(test_x, full_cov=True, size=3)
        a=6




        return posteriorTestY#samp# mu#mu_s#test_y #mu_s # predictions #mu #not y dummy

    def fit_model(self, train_x, train_y):


        self.train_x=train_x
        self.train_y=train_y

        data_indices = np.random.choice(self.train_x.shape[0], 100)
        #
        self.train_x = self.train_x[data_indices]
        self.train_y = self.train_y[data_indices]

        N=len(train_x)

        # K = kernel(train_x, train_x)
        # self.L = np.linalg.cholesky(K + self.s * np.eye(N))








        #= -0.5 * (np.dot(r, np.linalg.solve(K, r)) + np.linalg.slogdet(K)[1])


        """
             TODO: enter your code here
        """
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)
    #print(prediction.shape)


if __name__ == "__main__":
    main()
