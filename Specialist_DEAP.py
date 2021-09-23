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
experiment_name = "Eval_test"
output_folder = 'outputs/'

# if evaluate is true we evaluate our best saved solution
evaluate = True


########### initializing game variables ##########

n_hidden_neurons = 10

npop = 10              # population size
generations = 5        # number of generations
early_stopping = 50     # stop if fitness hasn't improved for x rounds   
dom_u = 1               # upper bound weight
dom_l = -1              # lower bound weight
tournament_size = 10    # individuals participating in tournament
mate_prob = 0.6         # crossover (mating) prob
mut_prob = 0.5          # mutation prob for individual
mut_gene_prob = 0.3     # mutation prob for each gene
mut_mu = 0              # mutation mean
mut_sigm = 1            # mutation sigma


enemies = [1,2]           # can be [1,2,3]
runs_per_enemy = 3     


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
                        randomini="yes",
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
eval_runs = 5
if evaluate:
    
    # run evaluation for every enemy
    for enemy, env in zip(enemies, envs):
        # env.update_parameter('speed','normal') # can be enabled if visualisation is True
        
        # make folder for overall enemy fitness
        exp_en_eval = output_folder + experiment_name + '_en' + str(enemy)
        if not os.path.exists(exp_en_eval):
            os.makedirs(exp_en_eval)

        bests_all_runs = []
        means_all_runs = []
                
        # go to run folder for every enemy
        for run in range(1, runs_per_enemy + 1):
            exp_en_run = exp_en_eval + "_run" + str(run)

            bests = []
            means = []

            # first load results to later calculate mean
            with open(exp_en_run + "/results.txt", "r") as results:
                next(results) #skip first line
                for line in results:
                    gen, best, mean, std, = line.split(' ')
                    bests.append(np.float(best))
                    means.append(np.float(mean))
            
            # loads best solution from that run
            best_sol = np.loadtxt(exp_en_run + "/best.txt")
            
            # evaluates it 'eval_runs' times and calculate mean
            eval_fits = []
            for eval_run in range(1, eval_runs+1):
                fit = evaluate_ind(best_sol, env)
                print("\nEnemy {0}, run {1}, eval run {2}: fitness = {3}".format(enemy, run, eval_run, fit))
                eval_fits.append(fit)
            eval_fits_mean = np.mean(eval_fits)

            # save mean evaluation of best solution in run folder
            file_aux  = open(exp_en_run +'/eval_fit.txt','w')
            print("\n ----- Mean fitness {0} of best solution saved for enemy {1}, run {2} -----".format(eval_fits_mean, enemy, run))
            file_aux.write(str(eval_fits_mean))
            file_aux.close()

            bests_all_runs.append(bests)
            means_all_runs.append(means)

        # calculate mean of best and mean per generation over each run
        mean_bests_all_runs = list(map(lambda x: sum(x)/len(x), zip(*bests_all_runs)))
        mean_means_all_runs = list(map(lambda x: sum(x)/len(x), zip(*means_all_runs)))

        # save mean of all bests per run
        file_aux  = open(exp_en_eval +'/bests.txt','w')
        file_aux.write(str(mean_bests_all_runs))
        file_aux.close()

        # save mean of all means per run
        file_aux  = open(exp_en_eval +'/means.txt','w')
        file_aux.write(str(mean_means_all_runs))
        file_aux.close()

    sys.exit()


########### initialing DEAP tools ##########

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

creator.create("FitnessMax", base.Fitness, weights = (1.0,))    # create maximization problem class
creator.create("Individual", list, fitness=creator.FitnessMax)  # link individual class to maximization problem class

tbx = base.Toolbox()
np.random.seed(123)
tbx.register("variable", np.random.uniform, dom_l, dom_u) # set variable instantiation function
tbx.register("individual", tools.initRepeat, creator.Individual, tbx.variable, n=n_vars) # set individual instantiation function
tbx.register("population", tools.initRepeat, list, tbx.individual) # set population instantiation function

# evolution properties
tbx.register("evaluate", evaluate_ind)
tbx.register("select", tools.selTournament, tournsize=tournament_size)
tbx.register("mate", tools.cxTwoPoint)
tbx.register("mutate", tools.mutGaussian, mu=mut_mu, sigma=mut_sigm, indpb=mut_gene_prob)


########### evolution ##########
# based on the DEAP documentation overview page: https://deap.readthedocs.io/en/master/overview.html

for enemy, env in zip(enemies, envs):

    for run in range(1, runs_per_enemy+1):

        # redefine output folder for every run
        exp_path_run = output_folder + experiment_name + '_en' + str(enemy) + '_run' + str(run)
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
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < mate_prob:
                    tbx.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < mut_prob:
                    tbx.mutate(mutant)
                    mutant.self = np.linalg.norm(mutant) # normalize
                    del mutant.fitness.values

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