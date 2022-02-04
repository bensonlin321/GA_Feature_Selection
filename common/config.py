from scipy.io import arff
import pandas as pd
import numpy as np
import os

# sklearn
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

SUCCESS_OUTPUT = {"status":True, "err_msg":"", "result":""}
FAIL_OUTPUT = {"status":False, "err_msg":"", "result":""}

GOD_CLASS = 1
DATA_CLASS = 2
FEATURE_ENVY = 3
LONG_METHOD = 4
SMELLS = ["","God Class", "Data Class", "Feature Envy", "Long Method"]

class Config:
    def __init__(self, strategy):
        self.X = False
        self.Y = False
        self.imputer = SimpleImputer(strategy=strategy)
        self.encoder = preprocessing.LabelEncoder()
        self.df = False
        self.arffdata = False

    def load_arff(self, filepath):
        try:
            #data = arff.loadarff(os.getcwd()+'/arff/feature-envy.arff')
            self.arffdata = arff.loadarff(filepath)
            self.df = pd.DataFrame(self.arffdata[0])
        except Exception as err:
            FAIL_OUTPUT["err"] = str(err)
            return FAIL_OUTPUT

    def _get_variables(self):
        try:
            d_res = self._get_dependent_variables()
            ind_res = self._get_independent_variables()
            if (not d_res["status"]):
                FAIL_OUTPUT["err_msg"] = d_res["err_msg"]
                return FAIL_OUTPUT

            if (not ind_res["status"]):
                FAIL_OUTPUT["err_msg"] = ind_res["err_msg"]
                return FAIL_OUTPUT
        except Exception as err:
            FAIL_OUTPUT["err_msg"] = str(err)
            return FAIL_OUTPUT

        return SUCCESS_OUTPUT

    def _get_dependent_variables(self):
        try:
            Y_d = self.df.iloc[:, -1].values
            self.Y = self.encoder.fit_transform(Y_d)
        except Exception as err:
            FAIL_OUTPUT["err_msg"] = str(err)
            return FAIL_OUTPUT
        return SUCCESS_OUTPUT

    def _get_independent_variables(self):
        try:
            X_d = self.df.iloc[:, :-1].copy()
            self.imputer.fit(X_d)
            self.X = self.imputer.transform(X_d)
        except Exception as err:
            FAIL_OUTPUT["err_msg"] = str(err)
            return FAIL_OUTPUT
        return SUCCESS_OUTPUT

    def read_x_y_data(self):
        try:
            res = self._get_variables()
            if (not res["status"]):
                FAIL_OUTPUT["err_msg"] = res["err_msg"]
                return FAIL_OUTPUT

            SUCCESS_OUTPUT["result"] = self.X, self.Y
        except Exception as err:
            FAIL_OUTPUT["err_msg"] = str(err)
            return FAIL_OUTPUT
        return SUCCESS_OUTPUT
