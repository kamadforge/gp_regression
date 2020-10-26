import numpy as np
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_classification

from sklearn.svm import SVC, LinearSVC
from time import time

from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.sparse import csc_matrix
from numpy.linalg import multi_dot
from numpy.linalg import norm

from math import pi
from  sklearn.metrics.pairwise import rbf_kernel

train_x_name = "train_x.csv"
train_y_name = "train_y.csv"

train_x = np.loadtxt(train_x_name, delimiter=',')
train_y = np.loadtxt(train_y_name, delimiter=',')

# load the test dateset
test_x_name = "test_x.csv"
test_x = np.loadtxt(test_x_name, delimiter=',')


def nystrom(X_train, X_test, gamma, c=500, k=200, seed=44):

    rng = np.random.RandomState(seed)
    n_samples = X_train.shape[0]
    idx = rng.choice(n_samples, c)

    X_train_idx = X_train[idx, :]
    W = rbf_kernel(X_train_idx, X_train_idx, gamma=gamma)

    u, s, vt = linalg.svd(W, full_matrices=False)
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k, :]

    M = np.dot(u, np.diag(1/np.sqrt(s)))

    C_train = rbf_kernel(X_train, X_train_idx, gamma=gamma)
    C_test = rbf_kernel(X_test, X_train_idx, gamma=gamma)

    X_new_train = np.dot(C_train, M)
    X_new_test = np.dot(C_test, M)

    return X_new_train, X_new_test


nystrom(train_x, train_x, 0.1)
