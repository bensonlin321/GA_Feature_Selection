from scipy.io import arff
import pandas as pd
import numpy as np
import os
import json
import csv
from common.config import *
np.set_printoptions(threshold=np.inf)

def check_cache_folder():
    file_path = "{}/cache".format(os.getcwd())
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)

def parse_multi_select(data):
    # Not support space
    if (' ' in data or '"' in data):
        print("[Error] Please follow the format and enter correct input")
        print("For example:[1,2] or 1,2")
        os._exit(1)

    try:
        data = json.loads(data)
    except Exception as err:
        try:
            data = data.split(",")
        except Exception as err:
            print("[Error] Please follow the format and enter correct input")
            print("For example:[1,2] or 1,2")
            os._exit(1)

    return data


def valid_input(data, msg):
    try:
        if(data == ''):
            print("[Error] Please enter {}".format(msg))
            return False
        try:
            data = int(data)
        except ValueError:
            print("[Error] Please enter a number")
            return False
        if data > 4 or data < 1:
            print("[Error] Please use correct {}".format(msg))
            return False
        return True
    except Exception as err:
        print("[Error] Can not parse data, please check input: {}".format(err))
        os._exit(1)

def valid_single_input(data):
    try:
        if(data == ''):
            print("[Error] Please enter a number")
            return False
        try:
            data = int(data)
        except ValueError:
            print("[Error] Please enter a number")
            return False
        if data > 2 or data < 1:
            print("[Error] Please use correct number")
            return False
        return True
    except Exception as err:
        print("[Error] Can not parse data, please check input: {}".format(err))
        os._exit(1)

def get_config(code_smell=FEATURE_ENVY):
    try:
        conf = Config(strategy="median")
        if code_smell == GOD_CLASS:
            file_path = os.getcwd()+'/arff/god-class.arff'
        elif code_smell == DATA_CLASS:
            file_path = os.getcwd()+'/arff/data-class.arff'
        elif code_smell == FEATURE_ENVY:
            file_path = os.getcwd()+'/arff/feature-envy.arff'
        elif code_smell == LONG_METHOD:
            file_path = os.getcwd()+'/arff/long-method.arff'
        else:
            file_path = os.getcwd()+'/arff/feature-envy.arff'

        conf.load_arff(file_path)
        res = conf.read_x_y_data()
        if (not res["status"]):
            print("Err:" + res["err_msg"])
            os._exit(1)
        return res["result"]

    except Exception as err:
        print("[Error] Can not read config file: {}".format(err))
        os._exit(1)

def load_cache(file_name):
    try:
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                res = {"result":(float(row['default_train']), float(row['default_train_optimized']),\
                       float(row['default_test']), float(row['default_test_optimized']),\
                       float(row['default_train_f1']), float(row['default_train_f1_optimized']),\
                       float(row['default_test_f1']), float(row['default_test_f1_optimized']))}
                return res
    except Exception as err:
        print("[Warning] Can not load cache: {}".format(err))
        return {}
