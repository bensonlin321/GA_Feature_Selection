

import time
import os
import numpy as np
import pandas as pd
from time import process_time_ns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine, load_breast_cancer

class Dataset:
    def __init__(self):
        self.target = []
        self.data = []

def load_wdbc_data(filename):
    data_set = Dataset()
    temp_target = np.loadtxt(filename, usecols=range(1,2), delimiter=',', dtype=str)
    data_set.data = np.loadtxt(filename, usecols=range(3,32), delimiter=',')

    target = []
    for t in temp_target:
        if t == 'M': # malignant
            target.append(1)
        else: # 'B' benign
            target.append(0)
    data_set.target = np.asarray(target)

    return data_set

def main():
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
    score = accuracy_score(Y_test, Y_pred)
    print("score:")
    print(score)

    t1_stop = process_time_ns() 

    print("Elapsed time during the whole program in seconds:", (t1_stop-t1_start)/100000000) 
    
if __name__ == "__main__":
    main()
