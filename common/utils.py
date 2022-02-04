from scipy.io import arff
import pandas as pd
import numpy as np
import os
import json
import csv
from common.config import *
np.set_printoptions(threshold=np.inf)

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
