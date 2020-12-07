import pickle
import numpy
import pandas
import os
from PathFiles import *

class Processor:

    def __init__(self):

        self.stateDatas = {}

    @staticmethod
    def load_pkl_data(pkl_filename):
        with open(pkl_filename, 'rb') as input:
            return pickle.load(input)

    def load_all_states(self):

        for filename in os.listdir(os.path.join(DATASET, OPT_DATA)):
            if ".pkl" in filename:
                self.stateDatas[filename.split(".")[0]] = self.load_pkl_data(os.path.join(DATASET,
                                                                                          OPT_DATA,
                                                                                          filename))

    def get_analysis_of(self, state, policy_details, log_scale=True):

        if len(self.stateDatas) == 0:
            self.load_all_states()

        if state in self.stateDatas:
            return self.stateDatas[state].get_projection_data(policy_detail=policy_details, log_scale=log_scale)
        else:
            return [], []

    def get_state_analysis_with_policy_list(self, state, policy_list, policy_data):

        name_map = {"mask": 'mask_mandate', "social_distance": 'social_distance',
                    "transit_stations": "transit_stations_percent_change_from_baseline",
                    "groc_pharma":"grocery_and_pharmacy_percent_change_from_baseline",
                    "retail_recreation":'retail_and_recreation_percent_change_from_baseline',
                    "sd_intent":"social_distance_intent", "mask_intent":"mask_intent",
                    "workplace":"workplaces_percent_change_from_baseline",
                    "parks":"parks_percent_change_from_baseline"}

        policy_dict = {}
        for polc, pold in zip(policy_list, policy_data):

            policy_dict[name_map[polc]] = pold

        log_d1, log_d2 = self.get_analysis_of(state, policy_dict)
        full_d1, full_d2 = self.get_analysis_of(state, policy_dict, False)

        return log_d1, log_d2, full_d1, full_d2




