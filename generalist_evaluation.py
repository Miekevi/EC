import os
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller

########### initializing file variables ##########

np.random.seed(1)  # set a seed so that the results are consistent for reviewers of code/results

# disable visuals for faster experiments
visuals = False
if not visuals:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# setting experiment name and creating folder for logs
experiment_name = "gen_1"
output_folder = 'outputs/'

# loads file with the best solution for testing
eval_simulation_runs = 5            # how many times each solution is tested on each enemy
eval_enemies = [1,2,3,4,5,6,7,8]    # which enemies are tested
number_of_groups = 2                # how many training groups are there for each experiments (exp1_gr1, exp1_gr2, etc...)
runs_per_group = 10                 # how many runs have been performed for each group


########### helper functions ##########

def simulation(x, env):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness


def evaluate_ind(x, env):
    x = np.array(x)
    return np.array(simulation(x, env))


def create_env(enemy):
    env = Environment(enemies=[enemy],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=player_controller(10),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini="yes",
                    timeexpire=1500,
                    visualmode="no",
                    savelogs="no")  # enabling this gives error because we log the output to 'outputs/<exp_name>'
    return env


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
    for run in range(1, runs_per_group + 1):
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
        for enemy_index, env in enumerate(eval_envs):
            eval_fits = []
            # for every evaluation run
            for eval_run in range(1, eval_simulation_runs + 1):
                fit = evaluate_ind(best_sol, env)
                # print("\nEnemy {0}, run {1}, eval run {2}: fitness = {3}".format(enemy_index+1, run, eval_run, fit))
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