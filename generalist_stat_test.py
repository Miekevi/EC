import numpy as np
import ast

from scipy.stats import kruskal

########### initializing script variables ##########

expnames = ["gen_1", "gen_2"]
group_enemies = [[2, 5, 6], [7, 8]]
a = []

# prints the Kruskal Wallis results per group, per EA to see if the means are significantly different
for group, enemies in enumerate(group_enemies):
    for i, expname in enumerate(expnames):
        means_gen_str = open("outputs/" + expname + "_gr" + str(group + 1) + "_eval/means_gen.txt", "r").readline()
        means_gen = np.array(ast.literal_eval(means_gen_str))[:generations_per_run]
        a.append(means_gen)

F, p = kruskal(a[0], a[1], a[2], a[3])
print(F)
print(p)
