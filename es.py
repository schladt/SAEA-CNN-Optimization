"""
Evolutionary Strategies Algorithm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import os

from util import *

# ES Hyperparameters
MU = 10 # initial population size
LAMBDA = MU * 2 # offspring population size
SIGMA_CROSSOVER_RATE = 1 # sigma crossover rate
X_CROSSOVER_RATE = 0.2 # x value crossover rate
SIGMA_MUTATION_RATE = 1 # sigma mutation rate
X_MUTATION_RATE = 1 # x value mutation rate
MAX_GENERATIONS = 10 # maximum number of generations
CONVERGE_THRESHOLD = 0.001 # threshold for convergence of generational diversity
FITNESS_ALPHA = 0.7 # fitness function alpha value
FITNESS_BETA = 0.001 # fitness function beta value
NUM_DIMENSIONS = 3 # number of hyperparameters to optimize

# CNN Hyperparameters
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
VALIDATION_TARGET = 95 # target validation accuracy
TRAIN_CONCURRENT = 6 # number of models to train concurrently
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
DATA_DIR = '~/data/'
DATASET = 'cifar'  # 'mnist' or 'cifar'

# ES Gnome class
class Genome():
    """
    ES Genome
    """

    def __init__(self, 
                 x=None, 
                 sigma=None,
                 num_dimensions=NUM_DIMENSIONS,
                 fitness_alpha=FITNESS_ALPHA,
                 fitness_beta=FITNESS_BETA,
                 data_dir=DATA_DIR,
                 dataset=DATASET,
                 valid_target=VALIDATION_TARGET,
                 batch_size=BATCH_SIZE,
                 num_epochs=NUM_EPOCHS,
                 ):
        """
        Initialize genome
        """
        self.x = x
        self.sigma = sigma
        self.num_dimensions = num_dimensions
        self.fitness_alpha = fitness_alpha
        self.fitness_beta = fitness_beta
        self.data_dir = data_dir
        self.dataset = dataset
        self.valid_target = valid_target
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.fitness = 0
        self.valid_acc = 0
        self.train_acc = 0
        self.num_epochs_trained = 0

        if x is None:
            self.x = np.zeros(num_dimensions)
            self.x[0] = np.random.uniform(0, 0.2) # learning rate
            self.x[1] = np.random.uniform(0, 1) # momentum
            self.x[2] = np.random.uniform(0, 0.001) # weight decay

        if sigma is None:
            self.sigma = np.zeros(num_dimensions)
            self.sigma[0] = self.x[0] / 10
            self.sigma[1] = self.x[1] / 10
            self.sigma[2] = self.x[2] / 10

        

    def get_fitness(self):
        """
        Calculate fitness as function of validation accuracy and number of batches trained
        """
        # Get dataloaders
        train_loader, valid_loader = get_loaders(data_dir=self.data_dir, dataset=self.dataset, batch_size=self.batch_size, download=False)
        model = SmallCNN().to(DEVICE)

        # SGD optimizer - momentum, weight decay are the hyperparameters we are optimizing
        optimizer = torch.optim.SGD(model.parameters(), lr=self.x[0], momentum=self.x[1], weight_decay=self.x[2])

        # Train model
        log_dict = train_model(model=model, 
                       num_epochs=self.num_epochs, 
                       optimizer=optimizer, 
                       device=DEVICE, 
                       train_loader=train_loader, 
                       valid_loader=valid_loader,
                       valid_target=self.valid_target,
                       print_=True)

        # Calculate fitness as a function of validation accuracy and number of batches trained
        valid_acc = log_dict['valid_acc_per_epoch'][-1]
        train_acc = log_dict['train_acc_per_epoch'][-1]
        num_epochs_trained = log_dict['num_epochs_trained']
        fitness = (self.fitness_alpha * valid_acc) /( self.fitness_beta * num_epochs_trained * self.batch_size)

        return fitness, valid_acc, train_acc, num_epochs_trained
        
    def __repr__(self):
        """
        Print genome
        """
        return f'x: {self.x}, fitness: {self.fitness}, sigma: {self.sigma}'

# ES Helper Functions    
def get_population_diversity(population):
    """ find generational diversity as the maximum distance between any two genomes in the population """ 
    max_dist = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if i < j:
                dist = np.linalg.norm(np.array(population[i].x) - np.array(population[j].x))
                if dist > max_dist:
                    max_dist = dist
    return max_dist

def mutate(self, mutation_rate=X_MUTATION_RATE):
    """
    Mutate genome by adding random values to genome
    """
    if np.random.uniform() < mutation_rate:
        for i in range(len(self.x)):
            self.x[i] = max((self.x[i] + np.random.normal(0, self.sigma[i])), 1e-6) # make sure all values are positive

    # self.fitness = self.get_fitness()

def crossover(self, other, crossover_rate=X_CROSSOVER_RATE):
    """
    Crossover two genomes by averaging their values
    """
    if np.random.uniform() < crossover_rate:
        for i in range(len(self.x)):
            self.x[i] = max(((self.x[i] + other.x[i]) / 2), 1e-6) # make all values are positive
            other.x[i] = self.x[i]

def sigma_mutate(self, mutation_rate=SIGMA_MUTATION_RATE):
    """
    Mutate Sigma using uncorrelated mutation with n-step size
    """
    if np.random.uniform() < mutation_rate:
        TAU_PRIME = 1 / np.sqrt(2 * self.num_dimensions)
        TAU = 1 / np.sqrt(2 * np.sqrt(self.num_dimensions))

        self.sigma = self.sigma * np.exp(TAU_PRIME * np.random.normal(0, 1) + TAU * np.random.normal(0, 1))

def sigma_crossover(self, other, crossover_rate=SIGMA_CROSSOVER_RATE):
    """
    Crossover Sigma by averaging their values
    """
    if np.random.uniform() < crossover_rate:
        self.sigma = (self.sigma + other.sigma) / 2
        other.sigma = self.sigma

def fitness(genome):
    """Helper function for multiprocessing of fitness calculation"""
    return genome.get_fitness()

def run_es(
    mu=MU,
    lambda_=LAMBDA,
    sigma_crossover_rate=SIGMA_CROSSOVER_RATE,
    x_crossover_rate=X_CROSSOVER_RATE,
    sigma_mutation_rate=SIGMA_MUTATION_RATE,
    x_mutation_rate=X_MUTATION_RATE,
    max_generations=MAX_GENERATIONS,
    converge_threshold=CONVERGE_THRESHOLD,
    display_stats=True
):
    """
    Run Evolutionary Strategies Algorithm
    """

    # Initialize population creation
    population = [Genome() for _ in range(mu)]

    # Update fitnesses using multiprocessing   
    x_values = [genome.x for genome in population]
    with ProcessPoolExecutor(TRAIN_CONCURRENT) as executor:
        results = list(tqdm(executor.map(fitness, population), total=len(population)))

    for i in range(len(population)):
        population[i].fitness = results[i][0]
        population[i].valid_acc = results[i][1]
        population[i].train_acc = results[i][2]
        population[i].num_epochs_trained = results[i][3]
    
    # Initialize lists to store generational statistics
    generational_max = []
    generational_min = []
    generational_mean = []
    generational_diversity = []

    # get statistics about initial generation
    fitnesses = [genome.fitness for genome in population]
    generational_max.append(np.max(fitnesses))
    generational_min.append(np.min(fitnesses))
    generational_mean.append(np.mean(fitnesses))
    generational_diversity.append(get_population_diversity(population))

    # loop through generations
    for gen_count in range(max_generations):
        # print generational stats
        if display_stats:
            print(f'ES {mu} {lambda_} {x_mutation_rate} {x_crossover_rate} {gen_count} {gen_count * lambda_} {generational_min[-1]:.4f} {generational_mean[-1]:.4f} {generational_diversity[-1]:.4f}')

        # container to store new generation
        new_population = []

        # loop to create new generation of size lambda_
        for i in range(lambda_ // 2):
            # Uniform random parent selection
            parent_1_idx = np.random.randint(0, mu)
            parent_2_idx = np.random.randint(0, mu)

            parent_1 = deepcopy(population[parent_1_idx])
            parent_2 = deepcopy(population[parent_2_idx])

            # sigma crossover (intermediate recombination)
            sigma_crossover(parent_1, parent_2, crossover_rate=sigma_crossover_rate)

            # sigma mutation 
            sigma_mutate(parent_1, mutation_rate=sigma_mutation_rate)
            sigma_mutate(parent_2, mutation_rate=sigma_mutation_rate)
            
            # x crossover intermediate recombination
            crossover(parent_1, parent_2, crossover_rate=x_crossover_rate)

            # x mutation
            mutate(parent_1, mutation_rate=x_mutation_rate)
            mutate(parent_2, mutation_rate=x_mutation_rate)
                
            # add to new generation
            new_population.append(parent_1)
            new_population.append(parent_2)

        # Update fitnesses using multiprocessing   
        x_values = [genome.x for genome in new_population]
        with ProcessPoolExecutor(TRAIN_CONCURRENT) as executor:
            results = list(tqdm(executor.map(fitness, new_population), total=len(new_population)))

        for i in range(len(new_population)):
            new_population[i].fitness = results[i][0]
            new_population[i].valid_acc = results[i][1]
            new_population[i].train_acc = results[i][2]
            new_population[i].num_epochs_trained = results[i][3]

        # select mu genomes to survive 
        # we are trying to maximize the fitness function so we take the highest mu fitnesses of parent and offspring
        population.extend(new_population)
        population.sort(key=lambda x: x.fitness, reverse=True)
        population = population[:mu]

        # get statistics about new population
        fitnesses = [genome.fitness for genome in population]
        generational_max.append(np.max(fitnesses))
        generational_min.append(np.min(fitnesses))
        generational_mean.append(np.mean(fitnesses))
        generational_diversity.append(get_population_diversity(population))

        # Check for termination conditions
        # (1) - terminate if generational diversity is below threshold
        if generational_diversity[-1] < converge_threshold :
            if display_stats:
                print(f'Population has converged with generational diversity below threshold of {converge_threshold }')
            break

        # (2) - terminate after max_generations (display message)
        if gen_count == max_generations - 1 and display_stats:
            print('The maximum number of generations has been reached')

    # evolution has finished, print final champion
    champion = population[np.argmax(fitnesses)]
    if display_stats:
        print(f'ES {mu} {lambda_} {x_mutation_rate} {x_crossover_rate} {gen_count} {gen_count * lambda_} {generational_min[-1]:.4f} {generational_mean[-1]:.4f} {generational_diversity[-1]:.4f}')
        print(f'Final Champion: {champion.x} Fitness: {champion.fitness}')

        # plot generational statistics
        plt.plot(generational_max, label='max fitness')
        plt.plot(generational_min, label='min fitness')
        plt.plot(generational_mean, label='mean fitness')
        plt.title('Generational Statistics')
        plt.legend()
        plt.show()

        # plot generational diversity
        plt.plot(generational_diversity, label='diversity')
        plt.title('Generational Diversity')
        plt.legend()
        plt.show()

    # save champion to file
    filename = os.path.join('results', f'es_{DATASET}_champion.txt')
    with open(filename, 'w') as f:
        f.write(f'{champion.x} {champion.fitness}')

    # Pickle generational statistics and save to file
    filename = os.path.join('results', f'es_{DATASET}_generational_stats.pkl')
    with open(filename, 'wb') as f:
        pickle.dump((generational_max, generational_min, generational_mean, generational_diversity), f)

                    

    return champion, (gen_count * lambda_)

if __name__ == '__main__':
    run_es()