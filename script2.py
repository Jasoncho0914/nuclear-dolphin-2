import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import math
from pomegranate import *
import csv
import json
import operator
from hmmlearn import hmm

#change if necessary
N_OBS = 100
N_STATES = 64

OBSERVATIONS_DATA_PATH = "./Observations_" + str(N_OBS) + ".csv"
LABELS_PATH = "./Label_" + str(N_STATES) + ".csv"
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
    labels = np.array(labels)
    observations = np.array(observations)
    for k in range(len(labels)):
        i = int(labels[k][0] - 1)
        j = int(labels[k][1] - 1)
        s = labels[k][4]
        l = d.setdefault(int(s), [])
        l.append(observations[i][j])
        d[int(s)] = l

    with open('frequency_' + str(N_OBS) + '_' + str(N_STATES) + '.json', 'w') as outfile:
        json.dump(d, outfile)

def calculateDistribution(observations, labels):
    print('Creating Distributions...')

    with open('frequency_' + str(N_OBS) + '_' + str(N_STATES) + '.json', 'r') as readfile:
        d = json.load(readfile)

    # for key, val in d.items():
    #     d2 = {}
    #     for i in range(N_OBS):
    #         d2[i] = 0

    #     total = float(len(val))
    #     for v in val:
    #         d2[int(v)] += 1

    #     for key2, val2 in d2.items():
    #         d2[key2] = d2[key2]/total

    #     d[key] = d2

    print len(d)

    dists = [None for _ in range(N_STATES)]
    d_filler = {}
    for i in range(N_OBS):
        d_filler[i] = 1.0/N_OBS
    for i in range(len(dists)):
        if str(i) in d:
            distribution = DiscreteDistribution(d[str(i)])
            # distribution = PoissonDistribution.from_samples(d[str(i)])
            dists[i] = distribution
        else:
            dists[i] = DiscreteDistribution(d_filler)
    print(str(len(dists)) + "\n")
    return dists


def createTransitionTable(observations, radius):
    print('Creating Transition Probabilities...\n')
    w = int(math.sqrt(N_STATES))
    transition_mat = np.array([[0.0]*N_STATES for _ in range(N_STATES)])
    grid = [[i + j for j in range(w)] for i in range(0, N_STATES, w)]

    for i in range(len(grid)):
        print grid[i]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            neighbors = neighboringStates(grid, i, j, radius)
            for s in neighbors:
                transition_mat[grid[i][j]][s] = 1.0/len(neighbors)
            # for s in range(len(transition_mat[grid[i][j]])):
            #     if s in neighbors:
            #         transition_mat[grid[i][j]][s] = 1.0/(len(neighbors)+1)
            #     else:
            #         transition_mat[grid[i][j]][s] = (1.0/(len(neighbors)+1))/(len(transition_mat[grid[i][j]]) - len(neighbors))
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

def generateStates(observations, labels):
    with open('frequency_' + str(N_OBS) + '_' + str(N_STATES) + '.json', 'r') as readfile:
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

    d_filler = {}
    for i in range(N_OBS):
        d_filler[i] = 1.0/N_OBS
    states = [State(None) for _ in range(N_STATES)]
    for i in range(len(states)):
        if str(i) in d:
            distribution = DiscreteDistribution(d[str(i)])
            states[i] = State(distribution)
        else:
            states[i] = State(DiscreteDistribution(d_filler))
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
    # findDistribution(observations, labels)
    # dists = calculateDistribution(observations, labels)
    trans_mat = createTransitionTable(observations, 2)
    starts = np.array([1.0/N_STATES for _ in range(N_STATES)])

    # starts = np.array([0.0 for _ in range(N_STATES)])
    # for i in range(50, 100, 10):
    #     for j in range(5, 10):
    #         starts[i+j] = 1.0/25.0
    # print starts


    print(len(trans_mat))
    print(len(trans_mat[0]))

    print('Running HMM...')
    X = np.array(observations)

    print("Part1")
    state_names = [str(i) for i in range(N_STATES)]

    states = generateStates(observations, labels)
    model = HiddenMarkovModel('example')
    model.add_states(states)

    a, b, p = format_transitions(trans_mat, states, model)
    model.add_transitions(a, b, p)
    model.bake()
    # model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, state_names=state_names)

    # res = []
    # for i in range(len(X)):
    #     sequence = [state.name for i, state in model.viterbi(X[i])[1]]
    #     res.append(np.array(sequence[1:]).astype(str))
    # writeCSV(np.array(res), 'sequences_' + str(N_OBS) + '_' + str(N_STATES) + '.csv', header=None)

    # data = model.to_json()
    # with open('model_' + str(N_OBS) + '_' + str(N_STATES) + '.json', 'w') as outfile:
    #     json.dump(data, outfile)

    # with open('model_' + str(N_OBS) + '_' + str(N_STATES) + '.json', 'r') as json_data:
    #     model = HiddenMarkovModel.from_json(json.load(json_data))

    print("Part2")
    seq = []
    end = 5
    skip = 2
    for r in range(0, end):
        seq.append([X[r][i] for i in range(0, len(X[r]), skip)])
    model.fit(seq, algorithm='baum-welch')

    print("Part3")
    output = [state.name for i, state in model.viterbi(seq[0])[1]]
    print(output[340/skip])
    print(output[500/skip])
    print(output[830/skip])
    print(output[920/skip])
    print(output[960/skip])
    print(output[970/skip])
    # print(output[808/skip])
    # print(output[404/skip])
    print('')

    output = [state.name for i, state in model.viterbi(seq[1])[1]]

    print(output[800/skip])
    print(output[920/skip])
    print(output[660/skip])
    print(output[640/skip])
    print(output[500/skip])

    # d = {}
    # for i in range(1,len(output)-1):
    #     d2 = d.setdefault(output[i], {})
    #     count = d2.setdefault(output[i+1], 0)
    #     d2[output[i+1]] = count + 1
    #     d[output[i]] = d2

    # for i in range(1,len(output2)-1):
    #     d2 = d.setdefault(output2[i], {})
    #     count = d2.setdefault(output2[i+1], 0)
    #     d2[output2[i+1]] = count + 1
    #     d[output2[i]] = d2

    # for i in range(1,len(output3)-1):
    #     d2 = d.setdefault(output3[i], {})
    #     count = d2.setdefault(output3[i+1], 0)
    #     d2[output3[i+1]] = count + 1
    #     d[output3[i]] = d2

    # print d[output[-1]]
    # print(model.viterbi(test)[1])

def createSubmission(labels):
    print('Creating submission...')

    sequences = np.array(pd.read_csv('sequences_' + str(N_OBS) + '_' + str(N_STATES) + '.csv', header=None))
    print(len(sequences))

    #preview
    # output = sequences[0]
    # print(output[340-1])
    # print(output[500-1])
    # print(output[830-1])
    # print(output[920-1])
    # print(output[960-1])
    # print(output[970-1])
    # print(output[808-1])
    # print(output[404-1])
    # print('')

    # output = sequences[1]
    # print(output[800-1])
    # print(output[920-1])
    # print(output[660-1])
    # print(output[640-1])
    # print(output[500-1])

    w = int(math.sqrt(N_STATES))

    max_x = max(labels.loc[:,2])
    min_x = min(labels.loc[:,2])
    max_y = max(labels.loc[:,3])
    min_y = min(labels.loc[:,3])
    x_unit = (max_x - min_x)/w
    y_unit = (max_y - min_y)/w

    mapping = {}
    for i in range(w):
        for j in range(w):
            x = min_x + x_unit*i + x_unit/2
            y = min_y + y_unit*j + y_unit/2
            s = i*w + j
            mapping[int(s)] = [x, y]

    d = {}
    for i in range(len(sequences)):
        for j in range(len(sequences[i])-1):
            d2 = d.setdefault(sequences[i][j], {})
            count = d2.setdefault(sequences[i][j+1], 0)
            d2[sequences[i][j+1]] = count + 1
            d[sequences[i][j]] = d2

    submission = [['id', 'value']]
    for i in range(6000, len(sequences)):
        if sequences[i][-1] in d:
            s = max(d[sequences[i][-1]].iteritems(), key=operator.itemgetter(1))[0]
            x, y = mapping[s]
        else:
            x, y = mapping[sequences[i][-1]]
        submission.append([str(i+1)+'x', x])
        submission.append([str(i+1)+'y', y])

    writeCSV(submission, "submission.csv", header=None)

def HMM2(observations, labels):
    print('Running HMM...')
    X = np.array(observations)

    model = hmm.GaussianHMM(n_components=64, n_iter=100)
    model.startprob_ = np.array([1.0/N_STATES for _ in range(N_STATES)])
    trans_mat = createTransitionTable(observations, 3)
    model.transmat_ = trans_mat
    print(trans_mat)
    n_train = 7
    seq = [[[v] for v in x] for x in X[:n_train]]
    lengths = [len(x) for x in seq]
    seq = np.concatenate(seq)
    model.fit(seq, lengths)  

    n_test = 7
    seq = [[[v] for v in x] for x in X[:n_test]]
    lengths = [len(x) for x in seq]
    seq = np.concatenate(seq)
    Z2 = model.predict(seq)
    res = [Z2[1000*i:1000*(i+1)] for i in range(n_test)]
    writeCSV(res, "res.csv", header=None)

    print model.transmat_

def createSubmission2(observations, labels):
    pass


if __name__ == "__main__":
    start = time.time()
    observations, labels = getData()
    # HMM(observations, labels)
    # createSubmission(labels)
    HMM2(observations, labels)
    print(time.time() - start)