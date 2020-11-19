import pandas as pd
import numpy as np
import pickle
import os
from PathFiles import *

class StateData:

    X_Labels = ['social_distance', 'social_distance_intent', 'mask_mandate', 'mask_intent',
                'retail_and_recreation_percent_change_from_baseline',
                'grocery_and_pharmacy_percent_change_from_baseline',
                'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
                'workplaces_percent_change_from_baseline']

    def __init__(self, state):
        self.state = state
        self.df = None
        self.X_data = None
        self.Y_Orig_data = None
        self.Y_data = None
        self.Y_start = None
        self.theta = None
        self.column_len = 0

    @staticmethod
    def data_projection(st, X, theta):
        proj = X * theta
        proj = proj.sum(axis=1)
        proj = pd.concat([st, proj]).reset_index(drop=True)
        proj = proj.cumsum()
        return proj

    def set_Y_orig_data(self):
        Y_Orig_data = pd.concat([self.Y_start, self.Y_data]).reset_index(drop=True)
        self.Y_Orig_data = Y_Orig_data.cumsum()

    def get_projection_data(self, policy_detail, log_scale=True):

        alt_X_data = self.X_data.copy()
        for key in policy_detail:
            if int(policy_detail[key]) > 0:
                alt_X_data[key] = float(policy_detail[key])/100

        Y_proj_data = self.data_projection(self.Y_start, alt_X_data, self.theta)
        if log_scale:
            return self.Y_Orig_data, Y_proj_data
        else:
            return np.exp(self.Y_Orig_data), np.exp(Y_proj_data)

    def dump(self):
        with open(os.path.join(DATASET, OPT_DATA, "{}.pkl".format(self.state)), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


