# Randomized Optimization
import mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_heartfailure = pd.read_csv ('./heart_failure_clinical_records_dataset.csv')
data_heartfailure = df_heartfailure[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

## Algorithms
def mimic(problem_fit, pop_size=200, keep_pct=0.2):
    start = time.perf_counter()
    best_state, best_fitness = mlrose.mimic(problem_fit, pop_size=pop_size,  keep_pct=keep_pct )
    end = time.perf_counter()
    time_to_run = end - start
    return best_fitness, time_to_run

def genetic(problem_fit, mutation_prob=0.1, pop_size=200):
    start = time.perf_counter()
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, pop_size=pop_size,  mutation_prob = mutation_prob)
    end = time.perf_counter()
    time_to_run = end - start
    return best_fitness, time_to_run

def simulated_annealing(problem_fit, max_iterations=40, na=None):
    start = time.perf_counter()
    best_state, best_fitness = mlrose.simulated_annealing(problem_fit, max_iters=max_iterations)
    end = time.perf_counter()
    time_to_run = end - start
    return best_fitness, time_to_run

def rand_hill_climb(problem_fit, max_iterations=40, restarts=5):
    start = time.perf_counter()
    best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_iters=max_iterations, restarts = restarts)
    end = time.perf_counter()
    time_to_run = end - start
    return best_fitness, time_to_run


## Optimization Problems
def getOneMaxProblemFit():
    fitness= mlrose.OneMax()
    return mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True)

def getTSPProblemFit():
    # city coordinates
    tsp_data = pd.read_csv ('./tsp.csv')
    tsp_array = np.array(tsp_data)
    # Create list of city coordinates
    coords_list = tsp_array
    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords = coords_list)
    return mlrose.TSPOpt(length = len(tsp_array), coords = coords_list, maximize=True)

def getSixPeakProblemFit():
    fitness = mlrose.SixPeaks(t_pct=0.15)
    return mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True)

problem_fit_1 = getOneMaxProblemFit()
problem_fit_2 = getTSPProblemFit()
problem_fit_3 = getSixPeakProblemFit()

## Neural Net
def runNeuralNet(algorithm):
    ## create test & train sets
    df = df_heartfailure
    n = len(df.columns)
    labels = df[df.columns[-1]]
    train, test, labels_train, labels_test = train_test_split(data_heartfailure, labels, test_size=0.2, stratify=labels)

    # setup neural network
    classifier = mlrose.NeuralNetwork(algorithm=algorithm)

    # train neural network
    start = time.perf_counter()
    classifier.fit(train, labels_train)
    end = time.perf_counter()
    time_to_train = end - start

    print("FITTED WEIGHTS")
    print(classifier.fitted_weights)

    # make predeiction with neural network
    start = time.perf_counter()
    classifier.predict(test)
    end = time.perf_counter()
    time_to_predict = end - start

    # score predictions
    score = classifier.score(test, labels_test)
    print(time_to_train)
    print(time_to_predict)
    print(score)
    return time_to_train, time_to_predict, score

## Helpers
# plot results
def plot_results(x, y, xLabel, yLabel, algName):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    chart_title = algName + ": " + yLabel + " vs " + xLabel
    ax.set_title(chart_title)
    plt.show()


def do_alg(alg, problem_fit, param1=None, param2 = None, in_range=range(1,100)):
    fitness = []
    times = []
    for i in in_range:
        best_fitness, time_to_run = alg(problem_fit, param1, param2)
        fitness.append(best_fitness)
        times.append(time_to_run)
    print("Avg Fitness")
    avg_fitness = np.average(fitness)
    print(avg_fitness)
    print("Time")
    print(np.average(times))

def divideByTen(val):
    return val / 10

def returnVal(val):
    return val

def do_alg_tweaked_params(alg, problem_fit, algName, paramName, isFirstParam = True, otherParamVal=None, param_range = range(1,100), paramFn=returnVal, max_its=100):
    params_all = []
    fitness_all = []
    time_all = []
    for param1 in param_range:
        param1 = paramFn(param1)
        print("param")
        print(param1)
        params_all.append(param1)
        fitness = []
        times = []
        for i in range(1, max_its):
            print('it')
            print(i)
            if isFirstParam == True:
                best_fitness, time_to_run = alg(problem_fit, param1, otherParamVal)
            else:
                best_fitness, time_to_run = alg(problem_fit, otherParamVal, param1)
            fitness.append(best_fitness)
            times.append(time_to_run)
        avg_fitness = np.average(fitness)
        fitness_all.append(avg_fitness)
        time_all.append(np.average(times))
    plot_results(params_all, fitness_all, paramName, "Best Fitness", algName)
    plot_results(params_all, time_all, paramName, "Time to Run", algName)


## Main Execution

## Random Hill
print("START RANDOM HILL ALGORITHM")
### One Max
print("Solving the One Max Problem")
do_alg(rand_hill_climb, problem_fit_1, 10,  5)
do_alg(rand_hill_climb, problem_fit_1, 10,  15)
do_alg(rand_hill_climb, problem_fit_1, 30,  5)
do_alg(rand_hill_climb, problem_fit_1, 30,  15)
do_alg(rand_hill_climb, problem_fit_1, 40,  30)

#### Graphs
do_alg_tweaked_params(rand_hill_climb, problem_fit_1, "Random Hill Climb", "Max Iterations", True, 5)
do_alg_tweaked_params(rand_hill_climb, problem_fit_1, "Random Hill Climb", "Random Restarts", False, 40)

### Traveling Sales Person
print("Solving the Traveling Sales Person Problem")
do_alg(rand_hill_climb, problem_fit_2, 40,  15)
do_alg(rand_hill_climb, problem_fit_2, 40,  30)
do_alg(rand_hill_climb, problem_fit_2, 60,  15)
do_alg(rand_hill_climb, problem_fit_2, 60,  30)

#### Graphs
do_alg_tweaked_params(rand_hill_climb, problem_fit_2, "Random Hill Climb", "Max Iterations", True, 5, range(1, 100), returnVal, 10)
do_alg_tweaked_params(rand_hill_climb, problem_fit_2, "Random Hill Climb", "Random Restarts", False, 40,  range(1, 100), returnVal, 10)

### Six Peaks
print("Solving the Six Peaks Problem")
do_alg(rand_hill_climb, problem_fit_3, 40,  15)
do_alg(rand_hill_climb, problem_fit_3, 40,  30)
do_alg(rand_hill_climb, problem_fit_3, 60,  15)
do_alg(rand_hill_climb, problem_fit_3, 60,  30)

#### Graphs
do_alg_tweaked_params(rand_hill_climb, problem_fit_3, "Random Hill Climb", "Max Iterations", True, 5, range(1, 100))
do_alg_tweaked_params(rand_hill_climb, problem_fit_3, "Random Hill Climb", "Random Restarts", False, 40,  range(1, 100))

## Simulated Annealing
print("START SIMULATED ANNEALING ALGORITHM")
### One Max
print("Solving the One Max Problem")
do_alg(simulated_annealing, problem_fit_1, 10,  None)
do_alg(simulated_annealing, problem_fit_1, 20,  None)
do_alg(simulated_annealing, problem_fit_1, 40,  None)
do_alg(simulated_annealing, problem_fit_1, 60,  None)

#### Graphs
do_alg_tweaked_params(simulated_annealing, problem_fit_1, "Simulated Annealing", "Max Iterations", True)

### Traveling Sales Person
print("Solving the Traveling Sales Person Problem")
do_alg(simulated_annealing, problem_fit_2, 10,  None)
do_alg(simulated_annealing, problem_fit_2, 20,  None)
do_alg(simulated_annealing, problem_fit_2, 40,  None)
do_alg(simulated_annealing, problem_fit_2, 60,  None)

#### Graphs
do_alg_tweaked_params(simulated_annealing, problem_fit_2, "Simulated Annealing", "Max Iterations", True)

### Six Peaks
print("Solving the Six Peaks Problem")
do_alg(simulated_annealing, problem_fit_3, 10, None)
do_alg(simulated_annealing, problem_fit_3, 20, None)
do_alg(simulated_annealing, problem_fit_3, 40, None)
do_alg(simulated_annealing, problem_fit_3, 60, None)

#### Graphs
do_alg_tweaked_params(simulated_annealing, problem_fit_3, "Simulated Annealing", "Max Iterations", True)

## Genetic Algorithm
print("START GENETIC ALGORITHM")
###  One Max
print("Solving the One Max Problem")
do_alg(genetic, problem_fit_1, 0.2,  25)
do_alg(genetic, problem_fit_1, 0.2,  50)
do_alg(genetic, problem_fit_1, 0.2,  100)
do_alg(genetic, problem_fit_1, 0.2,  150)

#### Graphs
do_alg_tweaked_params(genetic, problem_fit_1, "Genetic Algorithm", "Population Size", False, 0.2, range(1,400), returnVal, 5)

### Traveling Sales Person
print("Solving the Traveling Sales Person Problem")
do_alg(genetic, problem_fit_2, 0.2,  25)
do_alg(genetic, problem_fit_2, 0.2,  50)
do_alg(genetic, problem_fit_2, 0.2,  100)
do_alg(genetic, problem_fit_2, 0.2,  150)

#### Graphs
do_alg_tweaked_params(genetic, problem_fit_2, "Genetic Algorithm", "Population Size", False, 0.2, range(1,200), returnVal, 5)

### Six Peaks
print("Solving the Six Peaks Problem")
do_alg(genetic, problem_fit_3, 0.2,  25)
do_alg(genetic, problem_fit_3, 0.2,  50)
do_alg(genetic, problem_fit_3, 0.2,  100)
do_alg(genetic, problem_fit_3, 0.2,  150)

#### Graphs
do_alg_tweaked_params(genetic, problem_fit_3, "Genetic Algorithm", "Population Size", False, 0.2, range(1, 200), returnVal, 15)

## MIMIC Algorithm
print("START MIMIC ALGORITHM")

### One Max
print("Solving the One Max Problem")
do_alg(mimic, problem_fit_1, 25,  0.2)
do_alg(mimic, problem_fit_1, 50,  0.2)
do_alg(mimic, problem_fit_1, 25,  0.4)
do_alg(mimic, problem_fit_1, 50,  0.4)

#### Graphs
do_alg_tweaked_params(mimic, problem_fit_1, "MIMIC Algorithm", "Population Size", True, 0.2,  range(5, 100, 5), returnVal, 2)
do_alg_tweaked_params(mimic, problem_fit_1, "MIMIC Algorithm", "Keep_pct", False, 25, range(1, 10), divideByTen, 2)

### Traveling Sales Person
print("Solving the Traveling Sales Person Problem")

do_alg(mimic, problem_fit_2, 25, 0.3)
do_alg(mimic, problem_fit_2, 50, 0.3)
do_alg(mimic, problem_fit_2, 75, 0.3)
do_alg(mimic, problem_fit_2, 100, 0.3)

#### Graphs
do_alg_tweaked_params(mimic, problem_fit_2, "MIMIC Algorithm", "Population Size", True, 0.3, range(5, 125, 5), returnVal, 2)

### Six Peaks
print("Solving the Six Peaks Problem")
do_alg(mimic, problem_fit_3, 25, 0.3)
do_alg(mimic, problem_fit_3, 50, 0.3)
do_alg(mimic, problem_fit_3, 75, 0.3)
do_alg(mimic, problem_fit_3, 100, 0.3)
do_alg(mimic, problem_fit_3, 150, 0.3)

#### Graphs
do_alg_tweaked_params(mimic, problem_fit_3, "MIMIC Algorithm", "Population Size", True, 0.3, range(50, 200, 5), returnVal, 2)

## Neural Net
runNeuralNet('random_hill_climb')
runNeuralNet('simulated_annealing')
runNeuralNet('genetic_alg')

