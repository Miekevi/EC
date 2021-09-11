import os
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools

########### initializing file variables ##########

# disable visuals for faster experiments
visuals = False
if not visuals:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# setting experiment name and creating folder for logs
experiment_name = "Specialist_DEAP_1"
output_path = 'outputs/'
if not os.path.exists(output_path + experiment_name):
    os.makedirs(output_path + experiment_name)

# if evaluate is true we evaluate our best saved solution
evaluate = False


########### initializing game variables ##########

n_hidden_neurons = 10

npop = 50           # population size
generations = 50    # number of generations
dom_u = 1           # upper bound weight
dom_l = -1          # lower bound weight


########### initializing game ##########

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                    enemies=[1],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    savelogs="no") # enabling this gives error because we log the output to 'outputs/<exp_name>'

# logs the environment state
env.state_to_log()


########### helper functions ##########

def simulation(env,x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness

def evaluate_ind(x):
    x = np.array(x)
    return np.array(simulation(env,x))

def evaluate_pop(x):
    x = np.array(x)
    return np.array(list(map(lambda y: simulation(env,y), x)))


########### if evaluate is true, only evaluate and exit ##########

# loads file with the best solution for testing
if evaluate:
    try:
        best_sol = np.loadtxt('outputs/'+experiment_name+'/best.txt')
        print('\n --------- Running best saved solution ---------- \n')
        env.update_parameter('speed','normal')
        evaluate([best_sol])
    except IOError:
        print('ERROR: Solution to be evaluated cannot be found!')
    finally:
        sys.exit(0)


########### initialing DEAP tools ##########

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

creator.create("FitnessMax", base.Fitness, weights = (1.0,))    # create maximization problem class
creator.create("Individual", list, fitness=creator.FitnessMax)  # link individual class to maximization problem class

tbx = base.Toolbox()
tbx.register("variable", np.random.uniform, dom_l, dom_u) # set variable instantiation function
tbx.register("individual", tools.initRepeat, creator.Individual, tbx.variable, n=n_vars) # set individual instantiation function
tbx.register("population", tools.initRepeat, list, tbx.individual) # set population instantiation function

# evolution properties
tbx.register("evaluate", evaluate_ind)
tbx.register("select", tools.selTournament, tournsize=3)
tbx.register("mate", tools.cxTwoPoint)
tbx.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

pop = tbx.population(n=npop) # instantiate population


########### evolution ##########
# based on the DEAP documentation overview page: https://deap.readthedocs.io/en/master/overview.html
CXPB, MUTPB = 0.5, 0.2

# Evaluate the entire population
print("\n ----- GENERATION 0 -----")
fitnesses = evaluate_pop(pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = [fit]

best_sol = np.max(fitnesses)
mean_sol = np.mean(fitnesses)
std_sol = np.std(fitnesses)
print("\nBest: {0}, Mean: {1}, std: {2}".format(best_sol, mean_sol, std_sol))

for i in range(1, generations+1):
    
    # Select the next generation individuals
    offspring = tbx.select(pop, len(pop)) 
    # Clone the selected individuals
    offspring = list(map(tbx.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < CXPB:
            tbx.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < MUTPB:
            tbx.mutate(mutant)
            del mutant.fitness.values

    print("\n ----- GENERATION {0} -----".format(i))
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = evaluate_pop(pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    best_fitness = np.max(fitnesses)
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    print("\nBest: {0}, Mean: {1}, std: {2}".format(best_fitness, mean_fitness, std_fitness))

print("\n ----- SIMULATION ENDED -----")