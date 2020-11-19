import pandas as pd
import numpy as np
from StateData import StateData
from PathFiles import *
import os


class StateOptimiser(StateData):


    _ALPHA_ = 0.1
    _ITER_ = 3000

    def __init__(self, state):
        StateData.__init__(self, state)

    def load_state_file(self):
        df = pd.read_csv(os.path.join(DATASET, STATE_DATA, "{}.csv".format(self.state)))
        df = df.drop('Unnamed: 0', axis=1)
        df = df.drop('adm1_name', axis=1)
        self.df = df

    def set_normalised_X_data(self):

        X_data = self.df[self.X_Labels]
        X_data[['social_distance', 'mask_mandate']] *= 100
        X_data = X_data / 100
        self.X_data = pd.concat([pd.Series(1, index=X_data.iloc[1:, :].index, name='theta_0'), X_data], axis=1)
        self.column_len = len(self.X_data.columns)

    def set_log_Y_data(self):

        self.Y_data = np.log(self.df["cum_confirmed_cases"])
        self.Y_start = self.Y_data.iloc[:1]
        self.Y_data = self.Y_data.diff()

    def remove_top_row_of_the_data(self):
        self.Y_data = self.Y_data.iloc[1:]
        self.X_data = self.X_data.iloc[1:, :]

    def remove_zero_data(self):
        data = pd.concat([self.X_data, self.Y_data.reindex(self.X_data.index)], axis=1)
        data = data[data["cum_confirmed_cases"] > 0]
        self.X_data = data.drop(["cum_confirmed_cases"], axis=1)
        self.Y_data = data["cum_confirmed_cases"]

    def get_initial_theta(self):
        theta = np.array([1., -0.01, -0.01, -0.01, -0.01, 0.05, 0.05, 0.05, 0.05, 0.05])
        # theta[0] = 100
        # theta *= 0.01
        return theta

    @staticmethod
    def hypothesis(theta, X):
        return theta * X

    @staticmethod
    def computeCost(X, y, theta):
        y1 = StateOptimiser.hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        return sum(np.sqrt((y1 - y) ** 2)) / (2 * len(X))

    @staticmethod
    def optimise(X, y, theta, alpha, i):
        J = []  # cost function in each iterations
        k = 0
        print(theta)
        while k < i:

            y1 = StateOptimiser.hypothesis(theta, X)
            y1 = np.sum(y1, axis=1)
            for c in range(0, len(X.columns)):
                theta[c] = theta[c] - alpha * (sum((y1 - y) * X.iloc[:, c]) / len(X))
            j = StateOptimiser.computeCost(X, y, theta)
            J.append(j)
            k += 1
        return J, theta

    def optimise_state_data(self):

        self.load_state_file()
        self.set_normalised_X_data()
        self.set_log_Y_data()
        self.remove_top_row_of_the_data()
        self.remove_zero_data()
        errs, self.theta = StateOptimiser.optimise(self.X_data, self.Y_data,
                                                   self.get_initial_theta(),
                                                   StateOptimiser._ALPHA_, StateOptimiser._ITER_)

        self.set_Y_orig_data()



