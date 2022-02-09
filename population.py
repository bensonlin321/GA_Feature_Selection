from individual import Individual
import random


class Population:
    """
        A class that describes a population of virtual individuals
    """

    def __init__(self, dataset, labels, pop_size, mutation_rate, model, test_size, random_state):
        self.population = []
        self.generations = 0
        self.mutation_rate = mutation_rate
        self.best_ind = None
        self.worst_ind = None
        self.finished = False
        self.perfect_score = 1
        self.max_fitness = 0.0
        self.average_fitness = 0.0
        self.mating_pool = []
        self.test_size = test_size
        self.random_state = random_state #apply random_state if you want to fix the result from train_test_split 

        for i in range(pop_size):
            ind = Individual(dataset, labels, pop_size, model, self.test_size, self.random_state)
            ind.calc_fitness()

            if ind.fitness > self.max_fitness:
                self.max_fitness = ind.fitness

            self.average_fitness += ind.fitness
            self.population.append(ind)

        self.average_fitness /= pop_size

    def print_population_status(self):
        print("\nPopulation " + str(self.generations))
        print("Average fitness: " + str(self.average_fitness))
        print("Best individual: " + str(self.best_ind.genes))
        print("Best individual fitness: " + str(self.best_ind.fitness))
        print("Worst individual fitness: " + str(self.worst_ind.fitness))
        print("Best individual precision: " + str(self.best_ind.test_res_set[0]))
        print("Worst individual precision: " + str(self.worst_ind.test_res_set[0]))
        print("Best individual recall: " + str(self.best_ind.test_res_set[1]))
        print("Worst individual recall: " + str(self.worst_ind.test_res_set[1]))
        print("Best individual f1-score: " + str(self.best_ind.test_res_set[2]))
        print("Worst individual f1-score: " + str(self.worst_ind.test_res_set[2]))

    def get_generations(self):
        return self.generations

    # Generate a mating pool according to the probability of each individual
    def natural_selection(self):
        # Implementation suggestion based on Lab 3:
        # Based on fitness, each member will get added to the mating pool a certain number of times
        # a higher fitness = more entries to mating pool = more likely to be picked as a parent
        # a lower fitness = fewer entries to mating pool = less likely to be picked as a parent
        self.mating_pool = []
        constant = 1000
        weight_threshold = 0.3

        # create the pool with all the individuals according to their probability (fitness)
        for index, ind in enumerate(self.population):

            #fitness = int( round(ind.fitness) )
            # normalize fitness score range: (0 ~ 1)
            # abs(100  - 100) / abs(100 - (-100)) = 200 / 200
            # abs(50   - 100) / abs(100 - (-100)) = 150 / 200
            # abs(-99  - 100) / abs(100 - (-100)) = 1 / 200
            # abs(-100 - 100) / abs(100 - (-100)) = 0 / 200

            if ( abs(self.best_ind.fitness - self.worst_ind.fitness) ) == 0:
                weight = 1
            else:
                weight = abs((ind.fitness - self.worst_ind.fitness) / abs(self.best_ind.fitness - self.worst_ind.fitness))

            if weight >= weight_threshold:
                new_distribution = [index for i in range( int( weight * constant) )]
                self.mating_pool.extend(new_distribution)

        if len(self.mating_pool) <= 0:
            for index, ind in enumerate(self.population):
                self.mating_pool.append(index)

    # Generate the new population based on the natural selection function
    def generate_new_population(self):
        population_len = len(self.population)
        mating_pool_len = len(self.mating_pool)

        new_population = []
        self.average_fitness = 0.0
        for i in range(population_len):
            i_partner_a = random.randint(0, mating_pool_len - 1)
            i_partner_b = random.randint(0, mating_pool_len - 1)

            i_partner_a = self.mating_pool[i_partner_a]
            i_partner_b = self.mating_pool[i_partner_b]

            partner_a = self.population[i_partner_a]
            partner_b = self.population[i_partner_b]

            child = partner_a.crossover(partner_b)
            child.mutate(self.mutation_rate)
            child.calc_fitness()

            self.average_fitness += child.fitness
            new_population.append(child)

        self.population = new_population
        self.generations += 1
        self.average_fitness /= len(new_population)

    # Compute/Identify the current "most fit" individual within the population
    def evaluate(self):
        best_fitness = float('-inf')
        worst_fitness = float('inf')

        for ind in self.population:
            if ind.fitness >= best_fitness:
                best_fitness = ind.fitness
                self.best_ind = ind

            if ind.fitness <= worst_fitness:
                worst_fitness = ind.fitness
                self.worst_ind = ind

        if best_fitness == self.perfect_score:
            # match the gene
            self.finished = True