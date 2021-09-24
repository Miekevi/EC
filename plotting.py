import os
import ast
import numpy as np
import matplotlib.pyplot as plt

########### initializing script variables ##########

# give plotting name, iplots will be saved in folder with same name
plotting_name = "Method1_test"

# give experiments that should be plotted, with corresponding color codes
expnames = ["Method1"]
colors_line = ["r"]
colors_std = ["#ffcccb"]

enemies = [1]               # experiments must have at least been done on these enemies
runs_per_enemy = 1          # experiments must have at least this amount of runs
generations_per_run = 100   # experiments must have at least this amount of generations


########### plotting script ##########

if not os.path.exists("plots/" + plotting_name):
        os.makedirs("plots/" + plotting_name)

# make a plot for every enemy
for enemy in enemies:

    group = []

    # include every experiment in 'expnames'
    for expname, color_line, color_std in zip(expnames, colors_line, colors_std):
        
        # first save line plot for bests, means and stds for every generation

        # load bests_gen, means_gen and stds_gen
        bests_gen_str = open("outputs/" + expname + "_en" + str(enemy) + "_eval/bests_gen.txt", "r").readline()
        bests_gen = np.array(ast.literal_eval(bests_gen_str))[:generations_per_run]
        means_gen_str = open("outputs/" + expname + "_en" + str(enemy) + "_eval/means_gen.txt", "r").readline()
        means_gen = np.array(ast.literal_eval(means_gen_str))[:generations_per_run]
        stds_gen_str = open("outputs/" + expname + "_en" + str(enemy) + "_eval/stds_gen.txt", "r").readline()
        stds_gen = np.array(ast.literal_eval(stds_gen_str))[:generations_per_run]
        
        plt.plot(means_gen, color=color_line, label="Mean " + str(expname))
        plt.plot(bests_gen, ':', color=color_line, label="Best " + str(expname))
        plt.legend()
        lower = means_gen-stds_gen
        upper = means_gen+stds_gen
        plt.fill_between(np.array(range(0, len(means_gen))), lower, upper, color=color_std)

        # load average for all runs for enemy in experiment
        bests_run_str = open("outputs/" + expname + "_en" + str(enemy) + "_eval/bests_run.txt", "r").readline()
        bests_run = ast.literal_eval(bests_run_str)
        group.append(bests_run)

    # make lineplot per enemy    
    plt.title("Results over all generations for experiments " + str(expnames))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend(loc = 'lower right')

    plt.savefig("plots/" + plotting_name + "/generations_en" + str(enemy) + ".png")
    plt.clf()

    # make boxplot per enemy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(group)
    plt.title("Mean fitness per EA over " + str(runs_per_enemy) + " runs (Enemy = " + str(enemy) + ")")
    plt.ylabel("Fitness")
    plt.xlabel("EA name")
    ax.set_xticklabels(expnames)
    fig.savefig("plots/" + plotting_name + "/boxplot_en" + str(enemy) + ".png")
    plt.clf()

