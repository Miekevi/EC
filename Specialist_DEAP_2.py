import os
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, algorithms

########### initializing file variables ##########

# disable visuals for faster experiments
visuals = False
if not visuals:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# setting experiment name and creating folder for logs
experiment_name = "Specialist_DEAP_2"
output_folder = 'outputs/'

# if evaluate is true we evaluate our best saved solution
evaluate = False


########### initializing game variables ##########
# This is the same EA with some differences from the first one: evolutionary mechanisms, parameter tuning,
# fitness function etc.

n_hidden_neurons = 10

npop = 10               # population size
generations = 5         # number of generations
early_stopping = 25     # stop if fitness hasn't improved for x rounds
dom_u = 1               # upper bound weight
dom_l = -1              # lower bound weight
CXPB = 0.5              # crossover (mating) prob
MUTPB = 0.2             # mutation prob

enemies = [1]           # can be [1,2,3]
runs_per_enemy = 10


########### initializing game(s) ##########

# initialize simulations in individual evolution mode, for every enemy.

envs = []
for enemy in enemies:
    env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        savelogs="no") # enabling this gives error because we log the output to 'outputs/<exp_name>'
    envs.append(env)

env = envs[0]


########### helper functions ##########

def simulation(x, env):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness

def evaluate_ind(x, env):
    x = np.array(x)
    return np.array(simulation(x, env))

def evaluate_pop(x, env):
    x = np.array(x)
    return np.array(list(map(lambda y: simulation(y, env), x)))


########### if evaluate is true, only evaluate and exit ##########

# loads file with the best solution for testing
if evaluate:
    for enemy, env in zip(enemies, envs):
        try:
            best_sol = np.loadtxt('outputs/'+experiment_name+'_'+str(enemy)+'/best.txt') # loads solution from outputs/{expname}_{enemy}/best.txt
            print('\n --------- Running best saved solution for enemy '+str(enemy)+' ---------- \n')
            env.update_parameter('speed','normal')
            evaluate_ind(best_sol, env)
        except IOError:
            print('ERROR: Solution to be evaluated for enemy {0} cannot be found!'.format(str(enemy)))
    sys.exit()


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
tbx.register("select", tools.selNSGA2) # different selection operator
tbx.register("mate", tools.cxUniform, indpb=0.2) # different crossover (the one mentioned in paper)
tbx.register("mutate", tools.mutShuffleIndexes, indpb=0.2) # different mutation operator, instead of gaussian


########### evolution ##########
# based on the DEAP documentation overview page: https://deap.readthedocs.io/en/master/overview.html

for enemy, env in zip(enemies, envs):

    # redefine output folder for every enemy
    exp_path = output_folder + experiment_name + '_en' + str(enemy)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    for run in range(1, runs_per_enemy+1):
        # redefine output folder for every run
        exp_path_run = exp_path + '_run' + str(run)
        if not os.path.exists(exp_path_run):
            os.makedirs(exp_path_run)

        print("\n ----- SIMULATING FOR ENEMY {0}, run {1} -----".format(enemy, run))
        print("Writing outputs to " + exp_path_run)

        # instantiate population
        pop = tbx.population(n=npop)
        best_fit_exp = 0
        rnds_not_improved = 0

        # Evaluate the entire population
        print("\n ----- GENERATION 0 -----")
        fitnesses = evaluate_pop(pop, env)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = [fit]

        best_fit = np.max(fitnesses)
        mean_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)

        # saves results for first population
        file_aux  = open(exp_path_run+'/results.txt','w')
        file_aux.write('gen best mean std')
        print("\nGeneration: {0}, Best: {1}, Mean: {2}, std: {3}".format("0", best_fit, mean_fit, std_fit))
        file_aux.write("\n0 " + str(round(best_fit,6)) + ' ' +  str(round(mean_fit,6)) + ' ' + str(round(std_fit,6)))
        file_aux.close()

        # save new best fitness if better than current
        if best_fit > best_fit_exp:
            best_idx = np.argmax(fitnesses) # returns index of maximum
            np.savetxt(exp_path_run+'/best.txt', pop[best_idx])

        for i in range(1, generations+1):

            # Select the next generation individuals
            offspring = tbx.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(tbx.clone, offspring))

            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, tbx, CXPB, MUTPB) # is this the same as what Max has done?

            print("\n ----- GENERATION {0} -----".format(i))

            # Evaluate newcomers to the population
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_fit = np.array(list(map(lambda x: tbx.evaluate(x, env), invalid_ind)))
            for ind, fit in zip(invalid_ind, invalid_fit):
                ind.fitness.values = [fit]

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # set new population fitness
            fitnesses = [ind.fitness.values for ind in pop]

            best_fit = np.max(fitnesses)
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)

            # saves results for each generation
            file_aux  = open(exp_path_run+'/results.txt','a')
            print("\nGeneration: {0}, Best: {1}, Mean: {2}, std: {3}".format(i, best_fit, mean_fit, std_fit))
            file_aux.write("\n" + str(i) + ' ' +  str(round(best_fit,6)) + ' ' +  str(round(mean_fit,6)) + ' ' + str(round(std_fit,6)))
            file_aux.close()


            # save new best fitness if better than current
            if best_fit > best_fit_exp:
                best_idx = np.argmax(fitnesses) # returns index of maximum
                np.savetxt(exp_path_run+'/best.txt', pop[best_idx])
                rnds_not_improved = 0
            else:
                rnds_not_improved += 1

            if rnds_not_improved == early_stopping:
                print("\n ##### Fitness has not improved in {0} rounds, stopping simulation...".format(early_stopping))
                break

    print("\n ----- SIMULATION FOR ENEMY {0} IS COMPLETED -----".format(enemy))