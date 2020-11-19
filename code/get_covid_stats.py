"""
Date: October 25, 2020

Get state-wise and county-wise Confirmed, Deaths, Recovered counts of COVID-19 cases.

Data source: https://github.com/CSSEGISandData/COVID-19
"""

import pandas as pd
import os
from argparse import ArgumentParser


def get_county_row(county_list, state_list, target_county, target_state):
    """

    :param county_list: list of counties from current file (str)
    :param state_list: list of states from current file (str)
    :param target_county: name of the county to be located (str)
    :param target_state: name of the state to be located (str)
    :return:  row index matching target_county and target_state (int)
    """
    count = 0
    for county, state in zip(county_list, state_list):
        if type(county) == str and county.lower() == target_county.lower() and state.lower() == target_state.lower():
            # print(county, state)
            return count
        count += 1


def get_state_row(state_list, target_state):
    """

    :param state_list: list of states from current file (str)
    :param target_state:  name of the state to be located (str)
    :return: row index matching target_state (int)
    """
    count = 0
    for state in state_list:
        if state.lower() == target_state.lower():
            return count
        count += 1


def get_countywise(county, state):
    """
    Creates a file named "{county}.csv" with confirmend, deaths and recovered count

    :param county: name of the desired county (str)
    :param state: name of the desired state (str)
    :return: None
    """
    # iterate through every file in the directory
    loc = "../dataset/daily_reports_jhu"
    output_file = county+".csv"

    with open(output_file, "w") as f:
        f.write("confirmed\tdeaths\trecovered\n")
        for file in sorted(os.listdir(loc)):
            print("Date: {}".format(file))
            data = pd.read_csv(loc+"/"+file, sep=",", encoding='utf-8', engine="python")
            if "Admin2" in data.head(0):
                row_idx = get_county_row(data["Admin2"], data["Province_State"], county, state)
                row = data.iloc[row_idx:row_idx+1, :].values
                f.write(str(row[0][7]) + "\t" + str(row[0][8]) + "\t" + str(row[0][9]) + "\n")

    f.close()


def get_state_wise(state):
    """
    Creates a file named "{state}.csv" with confirmend, deaths and recovered count

    :param state:  name of the desired state (str)
    :return: None
    """
    loc = "../dataset/us_daily_report_statewise"
    output_file = state+".csv"

    with open(output_file, "w") as f:
        f.write("confirmed\tdeath\trecovered\n")
        for file in sorted(os.listdir(loc)):
            print("Date: {}".format(file))
            data = pd.read_csv(loc+"/"+file, sep=",", encoding="utf-8", engine="python")
            row_idx = get_state_row(data["Province_State"], state)
            row = data.iloc[row_idx:row_idx+1, :].values
            f.write(str(row[0][5])+"\t"+str(row[0][6])+"\t"+str(row[0][7])+"\n")
    f.close()


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("-c", "--county", help="Name of the valid US county. Must be used with -s flag.")
    parse.add_argument("-s", "--state", help="Name of the valid US state")

    args = vars(parse.parse_args())

    if args['county'] and args['state']:
        get_countywise(args['county'], args['state'])
    if args['state']:
        print(args['state'])
        get_state_wise(args['state'])
    else:
        print("Execute program with valid flags")
