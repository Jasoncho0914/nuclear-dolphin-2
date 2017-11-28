import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import math
from pomegranate import *

#change if necessary
OBSERVATIONS_DATA_PATH = "./Observations.csv"
LABELS_PATH = "./Label_100.csv"
OFFSET = 0.00001

def getData():
    print('Retreiving data...\n')
    observations = pd.read_csv(OBSERVATIONS_DATA_PATH, header=None)
    labels = pd.read_csv(LABELS_PATH, header=None)
    return observations, labels

def findDistribution(observations, labels):
    print('Finding initial distributions...\n')
    d = {}
    for k in range(labels.shape[0]):
        i = labels.loc[k, 0] - 1
        j = labels.loc[k, 1] - 1
    #     s = labels.loc[k, 4]
    #     l = d.setdefault(s, [])
    #     l.append(observations.loc[i, j])
    #     d[s] = l

    # for key, val in d.items()[0:10]:
    #     print(key)
    #     print(val)

def HMM(observations):
    X = np.array(observations)
    dist = []
    model = HiddenMarkovModel.from_samples(dist, n_components=10000, X=X) #change n_components based on label
    #fit with model.fit(sequences, algorithm='baum-welch')
    #then predict with model.predict(sequence, algorithm='viterbi')?
    #and see what it pukes out 

if __name__ == "__main__":
    start = time.time()
    observations, labels = getData()
    findDistribution(observations, labels)

    print(time.time() - start)