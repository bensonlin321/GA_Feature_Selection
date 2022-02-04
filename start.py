
from population import Population
import time
import matplotlib.pyplot as plt
import numpy as np
from time import process_time_ns
from common.utils import *
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os
from sklearn.datasets import load_wine, load_breast_cancer

TOTAL_TIMES = 1
MAX_NUM_OF_GEN = 100

class Dataset:
    def __init__(self):
        self.target = []
        self.data = []

def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

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

def sort_func(obj):
  return obj.fitness

def main():
    test_combination = [(200, 0.2)]
    #dataset, labels = get_config(FEATURE_ENVY)
    data = load_wdbc_data('wdbc.data')
    dataset = data.data
    labels = data.target
    
    model = svm.SVC(kernel='rbf')

    t1_start = process_time_ns() 
    for comb in test_combination:
        pop_size, mutation_rate = comb

        best_ind_record = []
        generation_record = []
        accumulate_generations = 0
        start_time = time.time()

        cnt = 0
        test_size = 0.15
        random_state = 42
        pop = Population(dataset, labels, pop_size, mutation_rate, model, test_size, random_state)
        pop.evaluate()

        while not pop.finished and cnt < MAX_NUM_OF_GEN:
            pop.natural_selection()
            pop.generate_new_population()
            pop.evaluate()
            pop.print_population_status()
            best_ind_record.append(pop.best_ind)
            cnt+=1

        generation_record.append(pop.get_generations())
        accumulate_generations += pop.get_generations()

        res = accumulate_generations

        final_best_ind = best_ind_record[0]
        for best_ind in best_ind_record:
            if best_ind.fitness > final_best_ind.fitness:
                final_best_ind = best_ind

        end_time = time.time()
        output_msg = "avg_gene: {}, exec time: {}".format(res, end_time - start_time)
        print("==============================")
        print("best genes:")
        print(final_best_ind.genes)
        print("best fitness:")
        print(final_best_ind.fitness)
        print(output_msg)
        print(generation_record)
        print("==============================")

        t1_stop = process_time_ns()

        print("Elapsed time during the whole program in nanoseconds:", t1_stop-t1_start) 
    
if __name__ == "__main__":
    main()
