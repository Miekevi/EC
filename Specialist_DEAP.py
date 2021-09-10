import os
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller


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

# set running mode ("train" or "test")
run_mode = "train"


########### initializing game variables ##########

n_hidden_neurons = 10

dom_u = 1           # upper bound weight
dom_l = -1          # lower bound weight
npop = 100          # population size
generations = 30    # number of generations
mutation = 0.2      # mutation constant
last_best = 0  


########### initializing game ##########

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                    enemies=[1],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    savelogs="no")

# logs the environment state
env.state_to_log()


########### helper functions ##########

def simulation(env,x):
    fitness, player_life, enemy_life, time = env.play(pcont=x)
    return fitness

def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# and some more...


########### if running mode is 'test', only evaluate and exit ##########

# loads file with the best solution for testing
if run_mode =='test':

    best_sol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([best_sol])

    sys.exit(0)


########### generating initial population ##########

# calculate number of variables
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
print("\nVariables: {0} weights per individual.".format(n_vars))

# generate pupulation with size=npop, variables=nvars, lower, upper bounds=dom_l,dom_u
pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
# runs simulation for every individual, saves list of fitnesses (n=npop) as fit_pop
fit_pop = evaluate(pop)

for i in range(generations):
    print(i)