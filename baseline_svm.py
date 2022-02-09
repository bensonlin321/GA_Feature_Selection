

import time
import os
import numpy as np
import pandas as pd
from time import process_time_ns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel
NUM_OF_FEAT = 11
FEATURE_NAME = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
                      "radius standard error", "texture standard error", "perimeter standard error", "area standard error", "smoothness standard error", "compactness standard error", "concavity standard error", "concave points standard error", "symmetry standard error", "fractal dimension standard error",
                      "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"]

np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_columns", None, 'display.expand_frame_repr', False)

class Dataset:
    def __init__(self):
        self.target = []
        self.data = []

def load_wdbc_data(filename):
    data_set = Dataset()
    temp_target = np.loadtxt(filename, usecols=range(1,2), delimiter=',', dtype=str)
    data_set.data = np.loadtxt(filename, usecols=range(2,32), delimiter=',')

    target = []
    for t in temp_target:
        if t == 'M': # malignant
            target.append(1)
        else: # 'B' benign
            target.append(0)
    data_set.target = np.asarray(target)

    return data_set

def testDefaultSVM():
    t1_start = process_time_ns()

    data = load_wdbc_data('wdbc.data')

    # get features and labels
    dataset = pd.DataFrame(data.data)
    labels = pd.Series(data.target)

    # prepare model
    model = svm.SVC(kernel='rbf')
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, 
                labels, test_size=0.15, stratify=labels)

    # predict
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    t1_stop = process_time_ns()
    score = accuracy_score(Y_test, Y_pred)
    test_res_set = precision_recall_fscore_support(Y_test, Y_pred, average='binary')

    # print result
    print("---------- Result for the test data set: ----------")
    msg = "precision: {}\nrecall: {}\nf1-score: {}\naccuracy_score: {}".format(test_res_set[0], test_res_set[1], test_res_set[2], score)
    print(msg)
    print("Elapsed time during the whole program in seconds:", (t1_stop-t1_start)/100000000, "sec") 
    print("---------------------------------------------------")

def testFeatureSelectionSVM():
    t1_start = process_time_ns()

    data = load_wdbc_data('wdbc.data')

    
    select_k_best_classifier = SelectKBest(chi2, k=NUM_OF_FEAT).fit(data.data, data.target)
    selected_feature_name = select_k_best_classifier.get_feature_names_out(FEATURE_NAME)
    X_new = select_k_best_classifier.fit_transform(data.data, data.target)

    # get features and labels
    dataset = pd.DataFrame(X_new, columns=selected_feature_name)
    labels = pd.Series(data.target)

    print(dataset)

    # prepare model
    model = svm.SVC(kernel='rbf')
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, 
                labels, test_size=0.15, stratify=labels)

    # predict
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    t1_stop = process_time_ns() 
    score = accuracy_score(Y_test, Y_pred)
    test_res_set = precision_recall_fscore_support(Y_test, Y_pred, average='binary')

    # print result
    print("---------- Result for the test data set: ----------")
    msg = "precision: {}\nrecall: {}\nf1-score: {}\naccuracy_score: {}".format(test_res_set[0], test_res_set[1], test_res_set[2], score)
    print(msg)
    print("Elapsed time during the whole program in seconds:", (t1_stop-t1_start)/100000000, "sec") 
    print("---------------------------------------------------")

def main():
    print("======= testDefaultSVM =======")
    testDefaultSVM()
    print("\n")
    print("======= testFeatureSelectionSVM =======")
    testFeatureSelectionSVM()
    
if __name__ == "__main__":
    main()
