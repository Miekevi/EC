import ast
import numpy as np

from scipy.stats import wilcoxon


########### initializing script variables ##########

expnames = ["spec_1", "spec_2"]

enemies = [3, 7, 8]         # experiments must have at least been done on these enemies
runs_per_enemy = 10         # experiments must have at least this amount of runs
generations_per_run = 50    # experiments must have at least this amount of generations

means_gen_1 = []
means_gen_2 = []

# prints the Wilcoxon results per enemy to see if the means are significantly different
# can be rewritten to give the test statistic over combined enemies

for enemy in enemies:

    for expname in expnames:
        means_gen_str = open("outputs/" + expname + "_en" + str(enemy) + "_eval/means_gen.txt", "r").readline()
        means_gen = np.array(ast.literal_eval(means_gen_str))[:generations_per_run]
        if expname == "spec_1":
            means_gen_1 = means_gen
        else:
            means_gen_2 = means_gen
    statistics = wilcoxon(means_gen_1,means_gen_2)
    print(enemy,statistics)

