import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import math
from pomegranate import *
import csv
import json
import cPickle as pickle

#change if necessary
OBSERVATIONS_DATA_PATH = "./Observations_1000.csv"
LABELS_PATH = "./Label_100.csv"
OFFSET = 0.00001

def getData():
    print('Retreiving data...\n')
    observations = pd.read_csv(OBSERVATIONS_DATA_PATH, header=None)
    labels = pd.read_csv(LABELS_PATH, header=None)
    return observations, labels

#Writes data into a csv file
def writeCSV(data, name, header=None):
    data = list(data)
    with open(name,'wb') as resultFile:
        wr = csv.writer(resultFile)
        if header:
            if type(header) == type([]):
                wr.writerow(header)
            else:
                wr.writerow([header])
            header = None
        
        for x in data:
            if type(x) == type([]) or type(x) == type(np.array([])):
                wr.writerow(list(x))
            else:
                wr.writerow([x])

def findDistribution(observations, labels):
    print('Finding initial distributions...\n')
    d = {}
    for k in range(labels.shape[0]):
        i = labels.loc[k, 0] - 1
        j = labels.loc[k, 1] - 1
        s = labels.loc[k, 4]
        l = d.setdefault(int(s), [])
        l.append(observations.loc[i, j])
        d[int(s)] = l

    with open('frequency3.json', 'w') as outfile:
        json.dump(d, outfile)

def calculateDistribution(observations, labels, n_states):
    n_obs = 1000 #based on number of categories for observations

    with open('frequency3.json', 'r') as readfile:
        d = json.load(readfile)

    for key, val in d.items():
        d2 = {}
        for i in range(n_obs):
            d2[i] = 0

        total = float(len(val))
        for v in val:
            d2[int(v)] += 1

        for key2, val2 in d2.items():
            d2[key2] = d2[key2]/total

        d[key] = d2


    dists = [None for _ in range(n_states)]
    d_filler = {}
    for i in range(n_obs):
        d_filler[i] = 1.0/n_obs
    for i in range(len(dists)):
        if str(i) in d:
            distribution = DiscreteDistribution(d[str(i)])
            dists[i] = distribution
        else:
            dists[i] = DiscreteDistribution(d_filler)
    return dists


def createTransitionTable(observations, radius, n_states):
    print('Creating Transition Probabilities...\n')
    w = int(math.sqrt(n_states))
    transition_mat = np.array([[0.0]*n_states for _ in range(n_states)])
    grid = [[i + j for j in range(w)] for i in range(0, n_states, w)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            neighbors = neighboringStates(grid, i, j, radius)
            for s in neighbors:
                transition_mat[grid[i][j]][s] = 1.0/len(neighbors)
    return transition_mat

def neighboringStates(grid, i, j, radius):
    l_bound = max(0, j - radius)
    r_bound = min(len(grid)-1, j+radius)
    t_bound = max(0, i - radius)
    b_bound = min(len(grid)-1, i+radius)

    states = []
    for i in range(t_bound, b_bound+1):
        for j in range(l_bound, r_bound+1):
            states.append(grid[i][j])
    return states

def generateStates(observations, labels, n_states):
    with open('frequency.json', 'r') as readfile:
        d = json.load(readfile)

    for key, val in d.items():
        d2 = {}
        total = float(len(val))
        for v in val:
            d2.setdefault(int(v), 0)
            d2[int(v)] += 1

        for key2, val2 in d2.items():
            d2[key2] = d2[key2]/total

        d[key] = d2

    states = [State(None) for _ in range(n_states)]
    for i in range(len(states)):
        if str(i) in d:
            distribution = DiscreteDistribution(d[str(i)])
            states[i] = State(distribution)
    return states

def format_transitions(trans_mat, states, model):
    a = []
    b = []
    p = []

    for i in range(len(states)):
        a.append(model.start)
        b.append(states[i])
        p.append(1.0/len(states))

    for i in range(len(trans_mat)):
        for j in range(len(trans_mat[i])):
            a.append(states[i])
            b.append(states[j])
            p.append(trans_mat[i][j])
    return a, b, p

def HMM(observations, labels):
    n_states = 100  #Change based on Label
    # findDistribution(observations, labels)
    dists = calculateDistribution(observations, labels, n_states)
    trans_mat = createTransitionTable(observations, 3, n_states)
    starts = np.array([1.0/n_states for _ in range(n_states)])

    print('Running HMM...')
    X = np.array(observations)

    print("Part1")
    # model = HiddenMarkovModel()

    # states = generateStates(observations, labels, n_states)

    # model.add_states(states)

    # a, b, p = format_transitions(trans_mat, states, model)

    # model.add_transitions(a, b, p)
    # model.bake(verbose=False)
    state_names = [str(i) for i in range(n_states)]
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, state_names=state_names)

    # data = model.to_json()
    # with open('model.txt', 'w') as outfile:
    #     json.dump(data, outfile)

    print("Part2")
    model.fit(X, algorithm='baum-welch')
    # model.fit(X, algorithm='baum-welch')

    print("Part3")
    output = [state.name for i, state in model.viterbi(X[0])[1]]
    print(output[340])
    print(output[808])
    print(output[991])
    print(output[734])
    print(output[247])
    print(output[719])
    print(output[317])
    print(output[404])


    res = []
    for i in range(n_states):
        res.append(np.array([str(state.name) for i, state in model.viterbi(X[i])[1]]))
    writeCSV(np.array(res), 'res2.csv', header=None)
    # print("Part3")
    # res = model.predict(X[0], algorithm='viterbi')
    # print res 

if __name__ == "__main__":
    start = time.time()
    observations, labels = getData()
    HMM(observations, labels)
    print(time.time() - start)