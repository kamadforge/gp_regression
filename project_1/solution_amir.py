import GPy
import scipy
import numpy as np


import matplotlib.pyplot as plt

# plt.switch_backend('MacOSX') # https://discourse.julialang.org/t/logout-on-mac-when-using-pyplot-v1-2/27842/7


BASELINE = False
LOCAL_COST_EVAL = False
NUM_SAMPLES = 500
if BASELINE:
    NUM_SAMPLES = 500

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

from random import seed

RANDOM_SEED = 12345
seed(RANDOM_SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted >= true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted >= THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted <= true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    reward = W4 * np.logical_and(predicted < THRESHOLD, true < THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        self.model = None
        pass

    def predict(self, test_x):

        if BASELINE:

            # PREDICTIONS_OFFSET = 0.00
            # PREDICTIONS_OFFSET = 0.05
            # PREDICTIONS_OFFSET = 0.10
            PREDICTIONS_OFFSET = 0.15  # gives 0.080 FOR NON-RANDOM SAMPLING of 1000 TRAINING POINTS
            # PREDICTIONS_OFFSET = 0.20
            # PREDICTIONS_OFFSET = 0.25

            return self.model.predict(test_x)[0].flatten() + PREDICTIONS_OFFSET

        # elif LOCAL_COST_EVAL:
        #         # FOR NON-DOCKER TESTING
        #         pred, cov = self.model.predict(test_x)
        #         TODO: pred.flatten()?
        #         return pred, cov

        else:
            pred, cov = self.model.predict(test_x)
            # LOW_CONF_OFFSET = 0.45
            # HIGH_CONF_OFFSET = 0.10
            LOW_CONF_OFFSET = 0.20
            HIGH_CONF_OFFSET = 0.10
            pred[cov >= 0.005] += LOW_CONF_OFFSET
            pred[cov < 0.005] += HIGH_CONF_OFFSET
            return pred.flatten()
        # else:
        #     pred, cov = self.model.predict(test_x)
        #     pred += 0.15
        #     # pred[np.logical_and(pred >= 0.50, cov >= .005)] += .15
        #     # pred[np.logical_and(pred < 0.35, cov >= .005)] += .15
        #     return pred.flatten()

    def fit_model(self, train_x, train_y):

        # needed for GPy
        train_y = np.expand_dims(train_y, axis=1)

        if BASELINE:
            train_x = train_x[:NUM_SAMPLES, :]
            train_y = train_y[:NUM_SAMPLES]
        else:
            # index = np.random.choice(train_x.shape[0], NUM_SAMPLES, replace=False)
            # train_x = train_x[index,:]
            # train_y = np.expand_dims(train_y, axis=1)[index]
            print(f'[INFO] creating balanced dataset.')
            # mins = scipy.spatial.distance.cdist(train_x, train_x)
            # mins = np.sort(mins, axis=1)
            # points_closest = mins[:, 3]
            # balanced_train_x_indices = np.where(points_closest > 0.0135)[0]
            cluster_1_train_x_indices = np.array((
                                                 5700, 5702, 5703, 5704, 5705, 5706, 5707, 5708, 5709, 5710, 5711, 5712,
                                                 5713, 5714, 5715, 5716, 5717, 5718, 5719, 5720, 5721, 5722, 5723, 5725,
                                                 5726, 5727, 5728, 5729, 5730, 5731, 5732, 5733, 5734, 5735, 5736, 5737,
                                                 5738, 5739, 5740, 5741, 5742, 5743, 5744, 5745, 5746, 5747, 5748, 5749,
                                                 11450, 11452, 11453, 11454, 11455, 11456, 11457, 11458, 11459, 11460,
                                                 11461, 11462, 11463, 11464, 11465, 11466, 11467, 11468, 11469, 11470,
                                                 11471, 11472, 11473, 11475, 11476, 11477, 11478, 11479, 11480, 11481,
                                                 11482, 11483, 11484, 11485, 11486, 11487, 11488, 11489, 11490, 11491,
                                                 11492, 11493, 11494, 11495, 11496, 11497, 11498, 11499, 17200, 17202,
                                                 17203, 17204, 17205, 17206, 17207, 17208, 17209, 17210, 17211, 17212,
                                                 17213, 17214, 17215, 17216, 17217, 17218, 17219, 17220, 17221, 17222,
                                                 17223, 17225, 17226, 17227, 17228, 17229, 17230, 17231, 17232, 17233,
                                                 17234, 17235, 17236, 17237, 17238, 17239, 17240, 17241, 17242, 17243,
                                                 17244, 17245, 17246, 17247, 17248, 17249))
            cluster_2_train_x_indices = np.random.choice(
                np.setdiff1d(np.array(range(17250)), cluster_1_train_x_indices),
                NUM_SAMPLES - cluster_1_train_x_indices.shape[0],
                replace=False,
            )
            balanced_train_x_indices = np.append(cluster_1_train_x_indices, cluster_2_train_x_indices)
            train_x = train_x[balanced_train_x_indices]
            train_y = train_y[balanced_train_x_indices]

        print(f'[INFO] fitting model on {train_x.shape[0]} samples.')

        # ----------------------------------------------------------------------
        #                                                         Select kernel
        # ----------------------------------------------------------------------
        # kernel = GPy.kern.RBF(input_dim=train_x.shape[1])
        # kernel = GPy.kern.RBF(input_dim=train_x.shape[1], ARD=True)
        # kernel = GPy.kern.Matern52(input_dim=train_x.shape[1], ARD=True)
        # kernel = GPy.kern.RBF(input_dim=train_x.shape[1], variance=1.0, lengthscale=1.0, ARD=True)
        kernel = GPy.kern.RBF(input_dim=train_x.shape[1], variance=1.0, lengthscale=1.0, ARD=True)

        # ----------------------------------------------------------------------
        #                                               Training sparse or not?
        # ----------------------------------------------------------------------
        model = GPy.models.GPRegression(train_x, train_y, kernel)
        # model = GPy.models.SparseGPRegression(train_x, train_y, kernel, num_inducing=5)
        # model = GPy.models.SparseGPRegression(train_x, train_y, kernel, num_inducing=10)

        # ----------------------------------------------------------------------
        #                                                      Fix some params?
        # ----------------------------------------------------------------------
        # fix gaussian noise term
        # model.rbf.lengthscale = 0.5
        # model.rbf.lengthscale.fix()
        # model.Gaussian_noise.variance = 0.01
        # model.Gaussian_noise.variance.fix()

        # ----------------------------------------------------------------------
        #                                                       Optimize params
        # ----------------------------------------------------------------------
        model.optimize_restarts(parallel=False, num_restarts=1)
        # model.optimize_restarts(parallel=False, num_restarts=3)
        # model.optimize_restarts(parallel=True, num_restarts=10)

        print(f'kernel lengthscale: {model.rbf.lengthscale.values[0]}')
        print(f'kernel variance: {model.rbf.variance.values[0]}')
        print(f'noise variance: {model.Gaussian_noise.variance.values[0]}')

        # ----------------------------------------------------------------------
        #                                                          Save visuals
        # ----------------------------------------------------------------------
        model.plot()
        plt.savefig(f'jigar.pdf')
        self.model = model


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

    if LOCAL_COST_EVAL:
        index = np.random.choice(train_x.shape[0], NUM_SAMPLES, replace=False)
        test_x = train_x[index, :]
        test_y = np.expand_dims(train_y, axis=1)[index]
        pred, cov = M.predict(test_x)
        for low_conf_offset in np.linspace(0, 1, 20, endpoint=False):
            for high_conf_offset in np.linspace(0, 1, 20, endpoint=False):
                tmp = pred.copy()
                tmp[cov >= 0.005] += low_conf_offset
                tmp[cov < 0.005] += high_conf_offset
                cost = cost_function(test_y, tmp)
                if cost < 0.05:
                    print(
                        f'Low conf offset: {low_conf_offset:.2f};\t High conf offset: {high_conf_offset:.2f};\t Cost: {cost}')


if __name__ == "__main__":
    main()