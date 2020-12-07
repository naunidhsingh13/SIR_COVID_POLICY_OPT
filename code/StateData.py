import pandas as pd
import numpy as np
import pickle
import os

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
        self.Y_data = None
        self.Y_Orig_start = None
        self.Y_Orig_data = None
        self.Y_Smooth_start = None
        self.Y_Smooth_data = None
        self.theta = None
        self.column_len = 0




