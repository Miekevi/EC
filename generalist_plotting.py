import os
import ast
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

########### initializing script variables ##########

# give plotting name, iplots will be saved in folder with same name
plotting_name = "task2_test"

# give experiments that should be plotted, with corresponding color codes
expnames = ["gen_4", "gen_4"]
colors_line = ["r", "b"]
colors_std = ["#ffcccb", "#add8e6"]


group_enemys = [[2, 5, 6], [7, 8]]
n_groups = len(group_enemys)        # number of of different groups of enemies trained on per experiment
runs_per_enemy = 10                 # experiments must have at least this amount of runs
generations_per_run = 50            # experiments must have at least this amount of generations
eval_enemies = [1,2,3,4,5,6,7,8]    # enemies used for evaluation


########### plotting script ##########

if not os.path.exists("plots/" + plotting_name):
        os.makedirs("plots/" + plotting_name)

# lineplots
for group, enemies in enumerate(group_enemys):

    for expname, color_line, color_std in zip(expnames, colors_line, colors_std):

        # load bests_gen, means_gen and stds_gen
        bests_gen_str = open("outputs/" + expname + "_gr" + str(group+1) + "_eval/bests_gen.txt", "r").readline()
        bests_gen = np.array(ast.literal_eval(bests_gen_str))[:generations_per_run]
        means_gen_str = open("outputs/" + expname + "_gr" + str(group+1) + "_eval/means_gen.txt", "r").readline()
        means_gen = np.array(ast.literal_eval(means_gen_str))[:generations_per_run]
        stds_gen_str = open("outputs/" + expname + "_gr" + str(group+1) + "_eval/stds_gen.txt", "r").readline()
        stds_gen = np.array(ast.literal_eval(stds_gen_str))[:generations_per_run]
        print(len(stds_gen))
        
        plt.plot(means_gen, color=color_line, label="Mean " + str(expname))
        plt.plot(bests_gen, ':', color=color_line, label="Best " + str(expname))
        lower = means_gen-stds_gen
        upper = means_gen+stds_gen
        plt.fill_between(np.array(range(0, len(means_gen))), lower, upper, color=color_std, alpha=0.4)

    # make lineplot per enemy    
    plt.title("Combined fitness over all generations for group {0}".format(enemies))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,100))
    plt.legend(loc = 'lower right')

    plt.savefig("plots/" + plotting_name + "/generations_gr" + str(group+1) + ".png")
    plt.clf()
    

# boxplots
for group, enemies in enumerate(group_enemys):

    for expname in expnames:

        # for every group-enemy combination make boxplot
        boxplot_df = pd.DataFrame()

        for enemy in eval_enemies:
            # read stats and add them together in one boxplot
            bests_run = open("outputs/" + expname + "_gr" + str(group+1) + "_eval/en" + str(enemy) + "_bests_run.txt", "r").readline()
            data = [float(f) for f in bests_run.strip('][').split(', ')]
            boxplot_df[str(enemy)] = data

        sns.boxplot(data=boxplot_df).get_figure()
        plt.title("{0} with training group {1}".format(expname, enemies))
        plt.xlabel("Enemy")
        plt.ylabel("Fitness")
        plt.ylim([0, 100])
        plt.savefig("plots/" + plotting_name + "/boxplots_" + str(expname) + "_gr" + str(group+1) + ".png")
        plt.clf()