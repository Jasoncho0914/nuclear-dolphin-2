import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import math


OBSERVATIONS_DATA_PATH = "./Observations.csv"
LABELS_PATH = "./Label.csv"
OFFSET = 0.00001

def getData():
    print('Retreiving data...\n')
    observations = pd.read_csv(OBSERVATIONS_DATA_PATH, header=None)
    labels = pd.read_csv(LABELS_PATH, header=None)
    return observations, labels

"""
    Discretizes observation values
    obs: observation data
    size: how many states to split it into
"""
def discretizeObs(obs, size=100):
    print('Discretizing observed states...\n')
    maxVal = obs.max().max()
    minVal = obs.min().min()
    
    unit = (maxVal - minVal)/size + OFFSET

    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            angle = obs.loc[i,j]
            s = round((angle-minVal)/unit)
            obs.loc[i,j] = s

    obs.to_csv("Observations_" + str(size) + ".csv", sep=',', index=False)

"""
    Discretizes hidden states (actual position)
    labels: labeled data
    size: how many states to split it into
"""
def discretizeHidden(labels, size=10000):
    print('Discretizing hidden states...\n')
    w = math.sqrt(size)
    if w != int(w):
        print("Invalid size, must be square (e.g 10 x 10 = 100)")
        return

    max_x = max(labels.loc[:,2])
    min_x = min(labels.loc[:,2])
    max_y = max(labels.loc[:,3])
    min_y = min(labels.loc[:,3])

    x_unit = (max_x - min_x)/w + OFFSET
    y_unit = (max_y - min_y)/w + OFFSET

    states = []

    for k in range(labels.shape[0]):
        x = labels.loc[k,2]
        y = labels.loc[k,3]

        i = round((x-min_x)/x_unit)
        j = round((y-min_y)/y_unit)

        states.append(int(i*w + j))

    df = pd.DataFrame({'state': states})

    labels = labels.join(df)
    labels.to_csv("Label_" + str(size) + ".csv", sep=',', index=False)

if __name__ == "__main__":
    start = time.time()
    observations, labels = getData()

    # discretizeHidden(labels, 100) #change 2nd parameter to adjust dicretization size
    # print(time.time() - start) #takes roughly 5 ~ 7 minutes

    discretizeObs(observations, 100) #change 2nd parameter to adjust dicretization size
    print(time.time() - start)
