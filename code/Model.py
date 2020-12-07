import numpy as np
import pandas as pd
import os
import pickle
from StateData import StateData
from PathFiles import *


class Model(StateData):

    def __init__(self, state):
        StateData.__init__(self, state)


    @staticmethod
    def model1(theta, X):
        return np.sum(theta * X, axis=1)

    @staticmethod
    def model2(theta, X):
        result = np.sum(theta[1:] * X.iloc[:, 1:], axis=1)+0.00001
        result = 1 + np.reciprocal(result)
        return np.maximum(np.zeros(result.shape[0]), theta[0] * np.reciprocal(result))

    @staticmethod
    def hypothesis(theta, X):
        X = X.copy()
        return Model.model2(theta, X)

    @staticmethod
    def data_projection(st, X, theta):
        proj = Model.hypothesis(theta, X)
        proj = pd.concat([st, proj]).reset_index(drop=True)
        proj = proj.cumsum()
        return proj[1:]

    def get_projection_data(self, policy_detail, log_scale=True):

        alt_X_data = self.X_data.copy()
        for key in policy_detail:
            if int(policy_detail[key]) > 0:
                alt_X_data[key] = float(policy_detail[key])/100

        Y_proj_data = self.data_projection(self.Y_Smooth_start, alt_X_data, self.theta)
        if log_scale:
            return self.Y_Orig_data, self.Y_Smooth_data, Y_proj_data
        else:
            return np.exp(self.Y_Orig_data), np.exp(self.Y_Smooth_data), np.exp(Y_proj_data)

    def dump(self):
        with open(os.path.join(DATASET, OPT_DATA, "{}.pkl".format(self.state)), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)