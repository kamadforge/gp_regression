import numpy as np
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

domain = np.array([[0, 5]])

from debug import ipsh

""" Solution """
# ACQUISITION_METHOD = 'UCB'
# ACQUISITION_METHOD = 'EI'
ACQUISITION_METHOD = 'EI_COMBINED'
ACQUISITION_EPS = 0.05
BUDGET = 10
V_OFFSET = 1.5

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.x_values = []
        self.f_values = []
        self.v_values = []

        kernel_f = 0.5 * Matern(length_scale=0.5, nu=2.5)
        kernel_v = np.sqrt(2) * Matern(length_scale=0.5, nu=2.5)
        self.model_f = GaussianProcessRegressor(kernel=kernel_f, alpha=0.15, random_state=0)
        self.model_v = GaussianProcessRegressor(kernel=kernel_v, alpha=0.0001, random_state=0)


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(
                objective,
                x0=x0,
                bounds=domain,
                approx_grad=True,
            )
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])


    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        def computeEI(mu_at_x, std_at_x, seen_obj_values):
            if std_at_x == 0:
                return 0
            else:
                if len(seen_obj_values) == 0:
                    best_obj = 0
                else:
                    best_obj = seen_obj_values[np.argmax(seen_obj_values)]
                z = (mu_at_x - best_obj - ACQUISITION_EPS) / std_at_x
                return (mu_at_x - best_obj - ACQUISITION_EPS) * norm.cdf(z) + std_at_x * norm.pdf(z)

        def computePI(mu_at_x, std_at_x, seen_obj_values):
            if std_at_x == 0:
                return 0
            else:
                best_obj = 1.2 - V_OFFSET
                z = (mu_at_x - best_obj - ACQUISITION_EPS) / std_at_x
                return norm.cdf(z)

        x_1 = np.reshape(x, (-1,1))
        mu_f, std_f = self.model_f.predict(x_1, return_std=True)
        mu_v, std_v = self.model_v.predict(x_1, return_std=True)
        ei_f = computeEI(mu_f, std_f, self.f_values)
        pi_v = computePI(mu_v, std_v, self.v_values)

        assert 0 <= pi_v <= 1, f'[ERROR] pi_v: {pi_v} not in range [0,1]'

        if ACQUISITION_METHOD == 'UCB':
            return mu_f + 2 * std_f
        elif ACQUISITION_METHOD == 'EI':
            return ei_f
        elif ACQUISITION_METHOD == 'EI_COMBINED':
            # return pi_v + ei_f
            # return pi_v * ei_f
            return 0 if pi_v < 0.5 else pi_v * ei_f
            return pi_v if pi_v < 0.5 else pi_v * ei_f


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        self.x_values.append(x.item())
        self.f_values.append(f.item())
        self.v_values.append(v.item() - V_OFFSET)

        self.model_f.fit(np.reshape(self.x_values, (-1,1)), np.array(self.f_values))
        self.model_v.fit(np.reshape(self.x_values, (-1,1)), np.array(self.v_values))


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        # iterate over all indicies where constraint is satisfied and find max objective
        max_objective = -1e3
        max_obj_index = -1

        for idx in np.argwhere(np.array(self.v_values) > 1.2 - V_OFFSET).flatten():
            if self.f_values[idx] > max_objective:
                max_objective = self.f_values[idx]
                max_obj_index = idx
        return self.x_values[max_obj_index]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    # kernel = np.sqrt(2) * Matern(length_scale=0.5, nu=2.5)
    # model_f = GaussianProcessRegressor(kernel=kernel, random_state=0)
    # output = model_f.predict(x)

    #"""Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return np.array(2.0)


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(BUDGET):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        print(f'Iteration #{j:02}/{BUDGET}: \t obj_val: {obj_val:.3f}, \t cost_val: {cost_val:.3f}')
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}')


if __name__ == "__main__":
    main()