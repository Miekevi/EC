import os
import sys
import numpy as np

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, algorithms

########### initializing file variables ##########

np.random.seed(1)  # set a seed so that the results are consistent for reviewers of code/results

# disable visuals for faster experiments
visuals = False
if not visuals:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# setting experiment name and creating folder for logs
experiment_name = "gen_4"
output_folder = 'outputs/'

# if evaluate is true we evaluate the results of an already ran experiment
# note that the data must be complete and all enemies must have the same amount of runs
evaluate = True

########### initializing game variables ##########
# This is a genetic algorithm, based on a combination of the specialist algorithms created for the first task

n_hidden_neurons = 10

npop = 100  # population size
generations = 50  # number of generations
early_stopping = 100  # stop if fitness hasn't improved for x rounds
dom_u = 1  # upper bound weight
dom_l = -1  # lower bound weight
tournament_size = 8    # individuals participating in tournament
# mate_prob = 0.5       # dynamic;
# mut_prob = 0.2        # dynamic;
mut_gene_prob = 0.4     # mutation prob for each gene

enemies_1 = [1, 2, 5]  # random
enemies_2 = [3, 4]  # random
runs_per_enemy = 10

########### initializing game(s) ##########

# initialize simulations in individual evolution mode, for every enemy.
if evaluate == False:

    envs = []

    env_1 = Environment(experiment_name=experiment_name,
                        enemies=enemies_1,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        randomini="yes",
                        timeexpire=1500,
                        visualmode="no",
                        savelogs="no")  # enabling this gives error because we log the output to 'outputs/<exp_name>'
    envs.append(env_1)

    env_2 = Environment(experiment_name=experiment_name,
                        enemies=enemies_2,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        randomini="yes",
                        timeexpire=1500,
                        visualmode="no",
                        savelogs="no")  # enabling this gives error because we log the output to 'outputs/<exp_name>'
    envs.append(env_2)

    env = envs[0]

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5



########### helper functions ##########

def simulation(x, env):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness


def evaluate_ind(x, env):
    x = np.array(x)
    return np.array(simulation(x, env))


def evaluate_pop(x, env):   # needs to get the average of all enemies, not doing it right now
    x = np.array(x)
    return np.array(list(map(lambda y: simulation(y, env), x)))


########### if evaluate is true, only evaluate and exit ##########

def create_env(enemy):
    env = Environment(enemies=[enemy],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini="yes",
                    timeexpire=1500,
                    visualmode="no",
                    savelogs="no")  # enabling this gives error because we log the output to 'outputs/<exp_name>'
    return env

# loads file with the best solution for testing
eval_simulation_runs = 5
eval_enemies = [1,2,3,4,5,6,7,8]
number_of_groups = 1

if evaluate:

    print("\n ----- Evaluation mode -----")

    eval_envs = [create_env(i) for i in eval_enemies]

    # run evaluation for every group
    for group in range(number_of_groups):
        # env.update_parameter('speed','normal') # can be enabled if visualisation is True

        # set experiment + enemy path for iteration
        exp_en = output_folder + experiment_name + '_gr' + str(group+1)

        # make folder for overall enemy analysis
        exp_en_eval = exp_en + "_eval"
        if not os.path.exists(exp_en_eval):
            os.makedirs(exp_en_eval)

        mean_bests_eval_enemies = [[] for _ in eval_enemies]
        
        bests_gen_all_runs = []
        means_gen_all_runs = []

        # go to run folder for every enemy
        for run in range(1, runs_per_enemy + 1):
            exp_en_run = exp_en + "_run" + str(run)

            bests_gen = []
            means_gen = []

            # first load results to later calculate mean
            with open(exp_en_run + "/results.txt", "r") as results:
                next(results)  # skip first line
                for line in results:
                    gen, best, mean, std, = line.split(' ')
                    bests_gen.append(np.float(best))
                    means_gen.append(np.float(mean))

            # loads best solution from that run
            best_sol = np.loadtxt(exp_en_run + "/best.txt")

            # evaluates it 'eval_runs' times, calculate mean and add it to the list
            for enemy_index, (enemy, env) in enumerate(zip(eval_enemies, eval_envs)):
                eval_fits = []
                # for every evaluation run
                for eval_run in range(1, eval_simulation_runs + 1):
                    fit = evaluate_ind(best_sol, env)
                    print("\nEnemy {0}, run {1}, eval run {2}: fitness = {3}".format(enemy+1, run, eval_run, fit))
                    eval_fits.append(fit)
                # mean fitness for a specific enemy
                eval_fits_mean = np.mean(eval_fits)
                mean_bests_eval_enemies[enemy_index].append(eval_fits_mean)

                bests_gen_all_runs.append(bests_gen)
                means_gen_all_runs.append(means_gen)

        # calculate mean of best and mean per generation over each run
        mean_bests_gen_all_runs = list(map(lambda x: sum(x) / len(x), zip(*bests_gen_all_runs)))
        mean_means_gen_all_runs = list(map(lambda x: sum(x) / len(x), zip(*means_gen_all_runs)))
        std_mean_means_gen_all_runs = list(map(lambda x: np.std(x), zip(*means_gen_all_runs)))

        # save the mean of all evaluations for every enemy
        for enemy_index, enemy in enumerate(eval_enemies):
            file_aux = open(exp_en_eval + '/en'+ str(enemy) + '_bests_run.txt', 'w')
            file_aux.write(str(mean_bests_eval_enemies[enemy_index]))
            file_aux.close() 

        # save mean of all bests per generation, averaged over runs
        file_aux = open(exp_en_eval + '/bests_gen.txt', 'w')
        file_aux.write(str(mean_bests_gen_all_runs))
        file_aux.close()

        # save mean of all means per generation, averaged over runs
        file_aux = open(exp_en_eval + '/means_gen.txt', 'w')
        file_aux.write(str(mean_means_gen_all_runs))
        file_aux.close()

        # save std of all means per generation, averaged over runs
        file_aux = open(exp_en_eval + '/stds_gen.txt', 'w')
        file_aux.write(str(std_mean_means_gen_all_runs))
        file_aux.close()

    sys.exit()


########### initialing DEAP tools ##########

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # create maximization problem class
creator.create("Individual", list, fitness=creator.FitnessMax)  # link individual class to maximization problem class

tbx = base.Toolbox()
tbx.register("variable", np.random.uniform, dom_l, dom_u)  # set variable instantiation function
tbx.register("individual", tools.initRepeat, creator.Individual, tbx.variable,
             n=n_vars)  # set individual instantiation function
tbx.register("population", tools.initRepeat, list, tbx.individual)  # set population instantiation function

# Evolution properties same as specialist_2
tbx.register("evaluate", evaluate_ind)
tbx.register("select1", tools.selRoulette)
tbx.register("select", tools.selTournament, tournsize=tournament_size)
tbx.register("mate", tools.cxUniform, indpb=0.5)
tbx.register("mutate", tools.mutShuffleIndexes, indpb=mut_gene_prob)

########### evolution ##########
# based on the DEAP documentation overview page: https://deap.readthedocs.io/en/master/overview.html

# for group, env in enumerate(envs):
if True:

    group = 1
    env = envs[1]

    for run in range(2, runs_per_enemy + 1):

        mate_prob = .15
        mut_prob = .85

        # Redefine output folder for every run
        exp_path_run = output_folder + experiment_name + '_gr' + str(group+1) + '_run' + str(run)
        if not os.path.exists(exp_path_run):
            os.makedirs(exp_path_run)

        print("\n ----- SIMULATING FOR GROUP {0}, run {1} -----".format(group+1, run))
        print("Writing outputs to " + exp_path_run)

        # instantiate population
        pop = tbx.population(n=npop)
        best_fit_exp = 0
        rnds_not_improved = 0

        # Evaluate the entire population
        print("\n ----- GENERATION 0 -----")
        fitnesses = evaluate_pop(pop, env)  # As calculated by cons_multi
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = [fit]

        best_fit = np.max(fitnesses)
        mean_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)

        # Saves results for first population
        file_aux = open(exp_path_run + '/results.txt', 'w')
        file_aux.write('gen best mean std')
        print("\nGeneration: {0}, Best: {1}, Mean: {2}, std: {3}".format("0", best_fit, mean_fit, std_fit))
        file_aux.write("\n0 " + str(round(best_fit, 6)) + ' ' + str(round(mean_fit, 6)) + ' ' + str(round(std_fit, 6)))
        file_aux.close()

        # save new best fitness if better than current
        if best_fit > best_fit_exp:
            best_idx = np.argmax(fitnesses)  # returns index of maximum
            np.savetxt(exp_path_run + '/best.txt', pop[best_idx])

        for i in range(1, generations + 1):

            mate_prob += .7/generations
            mut_prob -= .7/generations

            # Select the next generation individuals using 2 types of selection operators
            offspring1 = tbx.select1(pop, len(pop))
            offspring = tbx.select(offspring1, len(pop))
            # Clone the selected individuals
            offspring = list(map(tbx.clone, offspring))

            # Apply crossover and mutation on the offspring
            # Dynamic as in specialist_1
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < mate_prob:
                    tbx.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < mut_prob:
                    tbx.mutate(mutant)
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
            file_aux = open(exp_path_run + '/results.txt', 'a')
            print("\nGeneration: {0}, Best: {1}, Mean: {2}, std: {3}".format(i, best_fit, mean_fit, std_fit))
            file_aux.write("\n" + str(i) + ' ' + str(round(best_fit, 6)) + ' ' + str(round(mean_fit, 6)) + ' ' + str(
                round(std_fit, 6)))
            file_aux.close()

            # save new best fitness if better than current
            if best_fit > best_fit_exp:
                best_idx = np.argmax(fitnesses)  # returns index of maximum
                np.savetxt(exp_path_run + '/best.txt', pop[best_idx])
                rnds_not_improved = 0
            else:
                rnds_not_improved += 1

            if rnds_not_improved == early_stopping:
                print("\n ##### Fitness has not improved in {0} rounds, stopping simulation...".format(early_stopping))
                break

    print("\n ----- SIMULATION FOR GROUP {0} IS COMPLETED -----".format(group+1))
