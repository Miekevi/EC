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

n_groups = 1                        # number of of different groups of enemies trained on per experiment
runs_per_enemy = 10                 # experiments must have at least this amount of runs
generations_per_run = 50            # experiments must have at least this amount of generations
eval_enemies = [1,2,3,4,5,6,7,8]    # enemies used for evaluation


########### plotting script ##########

if not os.path.exists("plots/" + plotting_name):
        os.makedirs("plots/" + plotting_name)

# make lineplot for every experiment for every group
for expname, color_line, color_std in zip(expnames, colors_line, colors_std):

    # plot fitness over time (generations)
    for group in range(1,n_groups+1):

        # load bests_gen, means_gen and stds_gen
        bests_gen_str = open("outputs/" + expname + "_gr" + str(group) + "_eval/bests_gen.txt", "r").readline()
        bests_gen = np.array(ast.literal_eval(bests_gen_str))[:generations_per_run+1]
        means_gen_str = open("outputs/" + expname + "_gr" + str(group) + "_eval/means_gen.txt", "r").readline()
        means_gen = np.array(ast.literal_eval(means_gen_str))[:generations_per_run+1]
        stds_gen_str = open("outputs/" + expname + "_gr" + str(group) + "_eval/stds_gen.txt", "r").readline()
        stds_gen = np.array(ast.literal_eval(stds_gen_str))[:generations_per_run+1]
        
        plt.plot(means_gen, color=color_line, label="Mean " + str(expname))
        plt.plot(bests_gen, ':', color=color_line, label="Best " + str(expname))
        lower = means_gen-stds_gen
        upper = means_gen+stds_gen
        plt.fill_between(np.array(range(0, len(means_gen))), lower, upper, color=color_std, alpha=0.4)

    # make lineplot per enemy    
    plt.title("Fitness over all generations")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,100))
    plt.legend(loc = 'lower right')

    plt.savefig("plots/" + plotting_name + "/generations_gr" + str(group) + ".png")
    plt.clf()


    boxplot_df = pd.DataFrame()

    for enemy in eval_enemies:
        # read stats and add them together in one boxplot
        bests_run = open("outputs/" + expname + "_gr" + str(group) + "_eval/en_" + str(enemy) + "_bests_run.txt", "r").readline()
        boxplot_df[str(enemy)] = bests_run

    sns.boxplot(x="Enemy", y="Fitness", data=pd.melt(boxplot_df))
    plt.savefig("plots/" + plotting_name + "/boxplots_gr" + str(group) + ".png")
    plt.clf()
    
