import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Model import Model
from PathFiles import *
import os


def get_gaussian_weights(mu, sigma, rng=(-5, 6)):

    def gauss(xi):
        nonlocal mu, sigma
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * ((xi - mu) / sigma) ** 2)

    weights = []
    for x in range(rng[0], rng[1]):
        weights.append(gauss(x))

    s = sum(weights)
    weights = [w/s for w in weights]
    return weights


class StateOptimiser(Model):

    _ALPHA_ = 0.5
    _ITER_ = 3000
    _DAY_RNG_ = 7
    _SIGMA_ = 1
    _INC_PERIOD_ = 14
    _T_ = 0

    def __init__(self, state):
        Model.__init__(self, state)

    def load_state_file(self):
        df = pd.read_csv(os.path.join(DATASET, STATE_DATA, "new_{}.csv".format(self.state)))
        # df = df.drop('Unnamed: 0', axis=1)
        df = df.drop('adm1_name', axis=1)
        self.df = df

    def plot_graph(self, df):
        # df = pd.DataFrame()
        x = range(df.shape[0])

        if "Series" in str(type(df)):
            plt.plot(x, df.values, label="Count")
        else:
            for col in df.columns:
                y = df[col].values
                plt.plot(x, y, label=col)

        plt.legend()
        plt.show()


    def gauss_smooth_df(self, df):
        # self.plot_graph(df)
        weights = get_gaussian_weights(0, self._SIGMA_, rng=[-self._DAY_RNG_, self._DAY_RNG_+1])
        # df_pad_head = pd.DataFrame([df.mean(axis=0)]*(len(weights)//2), columns=df.columns)
        # df_pad_tail = pd.DataFrame([df.mean(axis=0)]*(len(weights)//2), columns=df.columns)
        df_pad_head = pd.DataFrame([df.iloc[0, :]] * (len(weights) // 2), columns=df.columns)
        df_pad_tail = pd.DataFrame([df.iloc[-1, :]] * (len(weights) // 2), columns=df.columns)
        df = pd.concat([df_pad_head, df, df_pad_tail], axis=0, join='outer', ignore_index=False)
        df = (df.rolling(window=len(weights), center=True).apply(lambda x: np.sum(weights * x), raw=False))
        df.dropna(inplace=True)
        # self.plot_graph(df)
        return df

    def gauss_smooth_series(self, s):
        # self.plot_graph(s)
        weights = get_gaussian_weights(0, self._SIGMA_, rng=[-self._DAY_RNG_, self._DAY_RNG_+1])
        # df_pad_head = pd.DataFrame([df.mean(axis=0)]*(len(weights)//2), columns=df.columns)
        # df_pad_tail = pd.DataFrame([df.mean(axis=0)]*(len(weights)//2), columns=df.columns)
        df_pad_head = pd.Series([s.iloc[0]] * (len(weights) // 2))
        df_pad_tail = pd.Series([s.iloc[-1]] * (len(weights) // 2))
        s = pd.concat([df_pad_head, s, df_pad_tail], axis=0, join='outer', ignore_index=False)
        s = (s.rolling(window=len(weights), center=True).apply(lambda x: np.sum(weights * x), raw=False))
        s.dropna(inplace=True)
        # self.plot_graph(s)
        return s


    def set_normalised_X_data(self):

        X_data = self.df[self.X_Labels]
        X_data[['social_distance', 'mask_mandate']] *= 100
        X_data = X_data / 100

        # Data transform :
        # X[0] -> 1 (Coeff of Theta_0)
        # X[1] to X[4] -> Policy Mandates : Inverse Relation ; p -> 1 - p
        # X[5] to X[9] -> Mobility : Re-Range : m -> m + 1
        X_data.iloc[:, 1:5] = 1 - X_data.iloc[:, 1:5]
        X_data.iloc[:, 5:9] = 1 + X_data.iloc[:, 5:]

        # Smoothing the input values with gaussian.
        X_data = self.gauss_smooth_df(X_data)

        # Adding column theta_0 with value 1
        self.X_data = pd.concat([pd.Series(1, index=X_data.iloc[1:, :].index, name='theta_0'), X_data], axis=1)
        self.column_len = len(self.X_data.columns)

    def set_log_Y_data(self):
        """
        Set three versions of Y_data (Cumulative Infected Count) -
        1) Log of Original,
        2) Smooth using Gaussian,
        3) Each day difference (of log). (to be passed to model)

        :return:
        """
        self.Y_Orig_data = np.log(self.df["cum_confirmed_cases"])
        self.Y_Smooth_data = self.gauss_smooth_series(self.Y_Orig_data)
        self.Y_data = self.Y_Smooth_data.diff()

    def align_X_Y(self):
        """
        Function to align X Data and associated Y Labels to match 14-15 days delta_T of virus incubation period.
        Also removing the first row, which transformed to Nan in Y_Label as diff was taken
        :return: Dataframe - X-Y data, which can be directly put to optimizer (The optimizers :) )
        """

        # Storing Y_start date to generate Y_data complete from prediction of diff from the model
        self.Y_Orig_start = self.Y_Orig_data.iloc[self._INC_PERIOD_:self._INC_PERIOD_+1]
        self.Y_Smooth_start = self.Y_Smooth_data.iloc[self._INC_PERIOD_:self._INC_PERIOD_+1]

        # Eliminating the first row - due to diff for each day taken, first element has to discarded.
        self.Y_data = self.Y_data.iloc[1:]
        self.Y_Orig_data = self.Y_Orig_data[1:]
        self.Y_Smooth_data = self.Y_Smooth_data[1:]
        self.X_data = self.X_data.iloc[1:, :]

        if self._INC_PERIOD_ > 0:
            self.X_data = self.X_data[:-self._INC_PERIOD_].reset_index(drop=True)
            self.Y_data = pd.Series(self.Y_data.iloc[self._INC_PERIOD_:].values)
            self.Y_Orig_data = pd.Series(self.Y_Orig_data.iloc[self._INC_PERIOD_:].values)
            self.Y_Smooth_data = pd.Series(self.Y_Smooth_data.iloc[self._INC_PERIOD_:].values)

    # def remove_zero_data(self):
    #     data = pd.concat([self.X_data, self.Y_data.reindex(self.X_data.index)], axis=1)
    #     data = data[data["cum_confirmed_cases"] > 0]
    #     self.X_data = data.drop(["cum_confirmed_cases"], axis=1)
    #     self.Y_data = data["cum_confirmed_cases"]

    def get_initial_theta(self):
        theta = np.array([1., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # theta[0] = 100
        # theta *= 0.01
        return theta


    @staticmethod
    def computeCost(X, y, theta):
        y1 = StateOptimiser.hypothesis(theta, X)
        # y1 = np.sum(y1, axis=1)
        return sum(np.sqrt((y1 - y) ** 2)) / (2 * len(X))

    @staticmethod
    def optimise(X, y, theta, alpha, i):
        J = []  # cost function in each iterations
        k = 0
        print(theta)
        while k < i:

            y1 = StateOptimiser.hypothesis(theta, X)
            for c in range(0, len(X.columns)):
                theta[c] = theta[c] - alpha * (sum((y1 - y) * X.iloc[:, c]) / len(X))
            j = StateOptimiser.computeCost(X, y, theta)
            J.append(j)
            print(j)
            k += 1
        return J, theta


    @staticmethod
    def optimise2(X, y, theta, alpha, i):
        J = []  # cost function in each iterations
        k = 0
        print(theta)
        while k < i:

            y1 = StateOptimiser.hypothesis(theta, X)

            a = np.sum(theta[1:]*X.iloc[:, 1:], axis=1)
            b = 1+np.reciprocal(a)

            theta_0_new = theta[0] - alpha * ((sum((y1 - y) *
                                                   (np.reciprocal(b))) / len(X)) -
                                              StateOptimiser._T_*np.exp(-StateOptimiser._T_*theta[0]))

            for c in range(1, len(X.columns)):
                theta[c] = theta[c] - alpha * ((sum((y1 - y) * theta[0] *
                                                   np.reciprocal(np.square(b)) *
                                                   np.reciprocal(np.square(a)) *
                                                   X.iloc[:, c]) / len(X)) -
                                               StateOptimiser._T_*np.exp(-StateOptimiser._T_*theta[c]))

            theta[0] = theta_0_new
            j = StateOptimiser.computeCost(X, y, theta)
            J.append(j)
            print(j)
            k += 1
        return J, theta

    def optimise_state_data(self):

        self.load_state_file()
        self.set_normalised_X_data()
        self.set_log_Y_data()
        self.align_X_Y()
        errs, self.theta = StateOptimiser.optimise2(self.X_data, self.Y_data,
                                                   self.get_initial_theta(),
                                                   StateOptimiser._ALPHA_, StateOptimiser._ITER_)




