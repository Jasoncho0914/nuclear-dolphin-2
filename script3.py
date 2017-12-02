import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import copy
from hmm import Hmm
import time

from discretizer import PointDiscretizer, AngleDiscretizer

# Observations: 10,000x1000 angles at each time step
# Labels: 600,000x4: run, step, x, y

# datapoints is an unsorted list
def bar_chart_countinuous(datapoints, interval, fname):
    plt.rcdefaults()

    labels = []
    values = []

    base = min(datapoints) - (min(datapoints) % interval)
    sum = 0
    for dp in datapoints:
        if dp < base+interval:
            sum += 1
        else:
            labels.append("{:.2f}-{:.2f}".format(base, base+interval))
            values.append(sum)
            sum = 0
            base = base + interval

    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.xticks(rotation=90)
    # plt.ylabel('Usage')
    plt.title('Distribution of Move Distances')

    plt.savefig(fname)



# counts is a map of integer values to counts
def bar_chart_discrete(counts, interval, fname):
    plt.rcdefaults()

    base = min(counts.keys()) - (min(counts.keys()) % interval)
    cap = (max(counts.keys()) - (max(counts.keys()) % interval)) + interval

    labels = []
    values = []

    for i in range(base, cap, interval):
        sum = 0
        for j in range(interval):
            index = i+j
            if index in counts:
                sum += counts[index]
        values.append(sum)
        labels.append("{}-{}".format(i, i+interval-1))



    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.xticks(rotation=90)
    # plt.ylabel('Usage')
    plt.title('Distances between consecutive labels')

    plt.savefig(fname)

def load_data_np():
    labels = np.loadtxt('Label.csv', delimiter=',')
    observations = np.loadtxt('Observations.csv', delimiter=',')
    return labels, observations

def load_data():
    labels, observations = load_data_np()
    labels = labels.tolist()
    for label in labels:
        label[0] = int(label[0])-1 # see below comment
        label[1] = int(label[1])-1 # minus one because they use base one indexing
    labels = [tuple(label) for label in labels]
    # sort labels to make sure they're in order with respect to runtime and step:
    labels.sort()
    observations = observations.tolist()
    return labels, observations

# generator for consecutive pairs of labels belonging to same run
# This code also ignores duplicates (stepdist 0)
def label_pairs(labels, max_stepdist = None):
    prev_l = labels[0]
    for next_l in labels[1:]:
        if prev_l[0] == next_l[0] and prev_l[1] != next_l[1]:
            if max_stepdist == None:
                yield prev_l, next_l
            elif next_l[1] - prev_l[1] <= max_stepdist:
                yield prev_l, next_l
        prev_l = next_l



def euclidean(tup):
    return math.sqrt( (tup[0] - tup[1])**2 + (tup[1] - tup[3])**2 )


def analysis(labels, observations):

    # determine the smallest and largest angle in 'observations', see what the scale is
    smallest_angle = math.inf
    largest_angle = -math.inf
    for run in observations:
        for angle in run:
            smallest_angle = min(smallest_angle, angle)
            largest_angle = max(largest_angle, angle)
    print("Smallest angle: {}".format(smallest_angle))
    print("Largest angle: {}".format(largest_angle))

    # determine the smallest and largest x, y value in labels:
    minx = math.inf
    maxx = -math.inf
    miny = math.inf
    maxy = -math.inf
    for label in labels:
        _, _, x, y = tuple(label)
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)
    print("X: {} - {}".format(minx, maxx))
    print("Y: {} - {}".format(miny, maxy))

    step_count = {}
    for p, s in label_pairs(labels):
        steps = int(round(abs(s[1] - p[1]), 0))
        if steps not in step_count:
            step_count[steps] = 1
        else:
            step_count[steps] += 1
    bar_chart_discrete(step_count, 3, 'lab_dist.png')
    # print("Smallest number of steps between consecutive observations: {}".format(min_steps))



    # Estimate the distance of an 'average' move using labels which are relatively close together:
    moves = []
    for p, s in label_pairs(labels, max_stepdist=1):
        stepdist = s[1] - p[1]
        try:
            moves.append(euclidean([p[2],p[3],s[2],s[3]]) / stepdist)
        except ZeroDivisionError:
            print("Error: {} {} {}".format(stepdist, p, s))
    print("Estimated Step Distances: Min {} Max {} Mean {} Std dev {}".format(min(moves), max(moves), statistics.mean(moves), statistics.stdev(moves)))
    bar_chart_countinuous(moves, .1, 'move_magnitude_dist.png')


def test_angle_discretizer():
    d = AngleDiscretizer(9)
    while True:
        s = input("theta in radians? ")
        x = float(s)
        print(d.discretize(x))


def test_point_discretizer():
    d = PointDiscretizer(10)
    while True:
        s = input("x,y?")
        x, y = tuple([float(f) for f in s.split(sep=',')])
        row, col = d.discretize(x,y)
        print("{},{}".format(row, col))
        print(d.un_discretize(row, col))

def smooth_transitions(transitions, pd, iterations, radius):
    shape = [pd.num_axis_subdivisions, pd.num_axis_subdivisions]

    trans = {}
    for from_point in transitions:
        trans[from_point] = np.zeros(shape)
        for to_point in transitions[from_point]:
            trans[from_point][to_point[0]][to_point[1]] = transitions[from_point][to_point]
    for _ in range(iterations):
        # for each probability table, smooth adjacent tiles:
        for origin_point in trans.keys():
            ptable = trans[origin_point]
            smooth_table = np.zeros(shape)
            # for each point on the table
            for r in range(shape[0]):
                for c in range(shape[1]):
                    if ptable[r][c] != 0:
                        # smooth over neighbors defined by 'radius'
                        adj_list = pd.adj(origin_point, radius)
                        addend = ptable[r][c] / len(adj_list)
                        for adj in adj_list:
                            smooth_table[adj[0]][adj[1]] += addend
            trans[origin_point] = ptable + smooth_table
    # finally, normalize
    for origin_point in trans.keys():
        ptable = trans[origin_point]
        sum = 0
        for r in range(shape[0]):
            for c in range(shape[1]):
                sum += ptable[r][c]
        for r in range(shape[0]):
            for c in range(shape[1]):
                ptable[r][c] = ptable[r][c] * 1/sum
        trans[origin_point] = ptable

    return trans


def load_hmm_data(pd, ad, labels, observations, smooth_iterations, smooth_radius, train_runs):
    # initialize start_probabilities to the uniform distribution
    start_probabilities = {}  # position state to probability
    for state in pd.states():
        start_probabilities[state] = 1 / pd.num_states

    # initialize transition probabilities using labels:
    # a) go through labels and count transitions:
    transitions = {}  # predecessor to successor to count
    for p, s in label_pairs(labels, max_stepdist=1):
        if p[0] >= train_runs: # ignore b.c. datapoint is reserved for validation
            continue
        p_state = pd.discretize(p[2], p[3])
        s_state = pd.discretize(s[2], s[3])
        if p_state not in transitions:
            transitions[p_state] = {}
        if s_state not in transitions[p_state]:
            transitions[p_state][s_state] = 1
        else:
            transitions[p_state][s_state] += 1
    transition_probabilities = smooth_transitions(transitions, pd, smooth_iterations, smooth_radius)  # position state to 2d np ndarray containing probabilities
    # finally, add uniform distributions for any states not visited:
    uniform_transition_p = np.zeros([pd.num_axis_subdivisions, pd.num_axis_subdivisions])
    for r in range(pd.num_axis_subdivisions):
        for c in range(pd.num_axis_subdivisions):
            uniform_transition_p[r][c] = 1/pd.num_states
    for r in range(pd.num_axis_subdivisions):
        for c in range(pd.num_axis_subdivisions):
            if (r,c,) not in transition_probabilities:
                transition_probabilities[(r,c,)] = uniform_transition_p

    # emission probabilities
    emission_probabilities = {}
    # iterate through all labels
    for label in labels:
        label_run = label[0]
        if label_run >= train_runs: # again, skip because this is validation data
            continue
        label_iteration = label[1]
        label_state = pd.discretize(label[2], label[3])

        if label_state not in emission_probabilities:
            emission_probabilities[label_state] = np.zeros([ad.num_states])
        disc_angle = ad.discretize(observations[label_run][label_iteration])
        emission_probabilities[label_state][disc_angle] += 1
    # finally, normalize emission probabilities:
    for state in emission_probabilities:
        sum = 0
        p_table = emission_probabilities[state]
        for i in range(p_table.shape[0]):
            sum += p_table[i]
        for i in range(p_table.shape[0]):
            p_table[i] = p_table[i] / sum
        emission_probabilities[state] = p_table

    return start_probabilities, transition_probabilities, emission_probabilities

def run_hmm(labels, observations, axis_divs, angle_divs, smooth_iterations, smooth_radius, validation_runs):
    train_runs = 6000-validation_runs

    pd = PointDiscretizer(axis_divs)
    ad = AngleDiscretizer(angle_divs)

    print("Loading HMM Data...", end='')
    start_time = time.time()
    start_probabilities, transition_probabilities, emission_probabilities = load_hmm_data(pd, ad, labels, observations, smooth_iterations, smooth_radius, train_runs)
    print("Done ({:.3f} secs)".format(time.time()-start_time))


    model = Hmm(start_probabilities, transition_probabilities, emission_probabilities)

    # grade the model on the validation runs
    # grading scheme: mean squared error (lower is better)
    print("Grading Model...", end='')
    start_time = time.time()
    hidden_states = pd.states()
    errors = []
    for validation_run in range(train_runs, 6000):
        run_labels = filter(lambda l : l[0] == validation_run, labels)
        observation_list = [ad.discretize(angle) for angle in observations[validation_run]]
        hidden_state_list = model.viterbi(observation_list, hidden_states)
        for label in run_labels:
            label_timestep = label[1]
            label_state = (label[2], label[3],)
            guessed_state = pd.un_discretize(hidden_state_list[label_timestep][0], hidden_state_list[label_timestep][1])
            errors.append(euclidean( guessed_state + label_state ))
    # get Mean Squared Error
    MSE = statistics.mean( map(lambda x: x*x, errors) )
    print("Done! ({}s)".format(time.time()-start_time))
    print("MSE of {:.3f} from AxDivs: {} AngDivs: {} SmIts: {} SmR: {} ValSet: {}".format(MSE, axis_divs, angle_divs, smooth_iterations, smooth_radius, validation_runs))
    return MSE

def localSearch(args, validation_runs):
    print("Loading Data...", end='')
    labels, observations = load_data()
    print("Done")

    index = -1
    prev_score = 100
    scores = []
    try:
        while True:
            index += 1
            index = index%len(args)
            # make a change:
            args[index] *= 2
            MSE = run_hmm(labels, observations, args[0], args[1], args[2], args[3], validation_runs)
            if MSE > prev_score: # undo change if no improvement
                args[index] = int(round(args[index]/2, 0))
            else:
                prev_score = MSE
            scores.append((MSE, args,))
    except KeyboardInterrupt:
        print("Writing all scores to file...")
        f = open('results.txt', 'w')
        scores.sort()
        for score in scores:
            f.write(str(score))
            f.write('/n')
        f.close()
        print("Done!")




def main():
    # run_hmm(10, 10, 5, 3, 10)
    # test_point_discretizer()
    localSearch([5, 64, 8, 8], 20)


    # labels, observations = load_data()
    # analysis(labels, observations)


if __name__ == '__main__':
    main()