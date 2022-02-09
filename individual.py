import random
import string
import numpy as np
import os
import pandas as pd

# sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
#np.set_printoptions(threshold=np.inf)
#pd.set_option("display.max_rows", None, "display.max_columns", None)

class Individual:
    """
        Individual in the population
    """

    def __init__(self, dataset, labels, pop_size, model, test_size, random_state):
        self.dataset = pd.DataFrame(dataset)
        self.labels = pd.Series(labels)
        self.fitness = 0
        self.pop_size = pop_size
        self.model = model
        self.X_train = False 
        self.X_test = False
        self.Y_train = False
        self.Y_test = False 
        self.test_size = test_size
        self.random_state = random_state
        self.genes = self.generate_random_genes()
        self.test_res_set = None

    # Fitness function: returns a floating points of "correct" characters
    def calc_fitness(self):

        # convert 0, 1 to true, false
        column_support = pd.Series(self.genes).astype(bool)

        # split dataset to testing data set and training data set
        #self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.dataset, 
        #    self.labels, test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.dataset, 
            self.labels, test_size=self.test_size, stratify=self.labels)


        # determine which feature that we want to use for calculating fitness score
        self.X_train_ = self.X_train[self.X_train.columns[column_support]]
        self.X_test_ = self.X_test[self.X_test.columns[column_support]]

        # predict and calculate score
        self.model.fit(self.X_train_, self.Y_train)
        Y_pred = self.model.predict(self.X_test_)
        score = accuracy_score(self.Y_test, Y_pred)
        test_res_set = precision_recall_fscore_support(self.Y_test, Y_pred, average='binary')

        self.fitness = score
        self.test_res_set = test_res_set


    #def __repr__(self):
    #    return ''.join(self.genes) + " -> fitness: " + str(self.fitness)

    def generate_random_genes(self):
        """
        Select the features from the individual
        
        Example:
        genes = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
        The 1 represents the presence of features and
        0 represents the absence of features
        
        """

        # generate random true or false for choosing the features
        genes = np.random.randint(2, size=len(self.dataset.iloc[0]))
        genes = genes.tolist()

        '''
        print("genes.count(1)")
        print(genes.count(1))
        print("len(self.dataset):")
        print(len(self.dataset))
        print("genes:")
        print(genes)
        '''
        return genes

    # The crossover function selects pairs of individuals to be mated, generating a third individual (child)
    def crossover(self, partner):
        # Crossover suggestion: child with half genes from one parent and half from the other parent
        ind_len = len(self.genes)
        child = Individual(self.dataset, self.labels, self.pop_size, self.model, self.test_size, self.random_state)

        midpoint = random.randint(0, ind_len)
        child.genes = self.genes[:midpoint] + partner.genes[midpoint:] # copy all the genes from genes until midpoint + copy all the partner genes start from midpoint to the end 

        return child

    # Mutation: based on a mutation probability, the function picks a new random character and replace a gene with it
    def mutate(self, mutation_rate):
        # code to mutate the individual here
        if random.uniform(0, 1) < mutation_rate:
            ind_len = len(self.genes)
            x = random.randint(0, ind_len-1)
            self.genes[x] ^= 1



