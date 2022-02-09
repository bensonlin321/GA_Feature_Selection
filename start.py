
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

MAX_NUM_OF_GEN  = 100
POPULATION_SIZE = 100
MUTATION_RATE   = 0.2
TEST_SIZE       = 0.15
RANDOM_STATE    = 42
FEATURES_NAME   = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
                   "radius standard error", "texture standard error", "perimeter standard error", "area standard error", "smoothness standard error", "compactness standard error", "concavity standard error", "concave points standard error", "symmetry standard error", "fractal dimension standard error",
                   "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"]

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
    data_set.data = np.loadtxt(filename, usecols=range(2,32), delimiter=',')

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
    test_combination = [(POPULATION_SIZE, MUTATION_RATE)]
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
        pop = Population(dataset, labels, pop_size, mutation_rate, model, TEST_SIZE, RANDOM_STATE)
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

        # prepare selected feature name
        selected_feature_name = []
        cnt = 0
        for gene in final_best_ind.genes:
            if gene == 1:
                selected_feature_name.append(FEATURES_NAME[cnt])
            cnt += 1

        print("==============================")
        print("Best genes:")
        print(final_best_ind.genes)
        print("Best selected features:")
        print(selected_feature_name)
        print("Best fitness:")
        print(final_best_ind.fitness)
        print("Best precision: " + str(final_best_ind.test_res_set[0]))
        print("Best recall: " + str(final_best_ind.test_res_set[1]))
        print("Best f1-score: " + str(final_best_ind.test_res_set[2]))
        print(output_msg)
        print(generation_record)
        print("==============================")

        t1_stop = process_time_ns()

        print("Elapsed time during the whole program in nanoseconds:", t1_stop-t1_start) 
    
if __name__ == "__main__":
    main()
