import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import math
from pomegranate import *
import csv

#change if necessary
OBSERVATIONS_DATA_PATH = "./Observations_100.csv"
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
    with open("./Scripts/Results/" + name + ".csv",'wb') as resultFile:
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
        l = d.setdefault(s, [])
        l.append(observations.loc[i, j])
        d[s] = l

    print (len(d))
    with open("Frequency.csv",'wb') as resultFile:
        wr = csv.writer(resultFile)
        for key, val in d.items():
            if type(val) == type([]):
                wr.writerow(list(val))

    # for key, val in d.items()[0:5]:
    #     d2 = {}
    #     for v in val:
    #         d2.setdefault(v, 0)
    #         d2[v] += 1

def HMM(observations, labels):
    print('Running HMM...\n')
    X = np.array(observations)
    print("Part1")
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=100, X=X) #change n_components based on label
    print("Part2")
    model.fit(X, algorithm='baum-welch')
    print("Part3")
    res = model.predict(X[0], algorithm='viterbi')
    print res 

if __name__ == "__main__":
    start = time.time()
    observations, labels = getData()
    HMM(observations, labels)
    print(time.time() - start)