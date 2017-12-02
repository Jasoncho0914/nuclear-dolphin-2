class Hmm:
    def __init__(self, start_p, hidden_transition_p, emission_p):
        self.start_p = start_p

        # self.min_start_p = 1
        # for prob in self.start_p.values():
        #     self.min_start_p = min(self.min_start_p, prob)

        self.hidden_transition_p = hidden_transition_p

        # self.min_transition_p = 1
        # for d in self.hidden_transition_p.values():
        #     for prob in d.values():
        #         self.min_transition_p = min(self.min_transition_p, prob)

        self.emission_p = emission_p

        # self.min_emission_p = 1
        # for d in self.emission_p.values():
        #     for prob in d.values():
        #         self.min_emission_p = min(self.min_emission_p, prob)

    def transition_probability(self, from_state, to_state):
        p = self.hidden_transition_p[from_state][to_state[0]][to_state[1]]
        if p == 0:
            return .0001
        else:
            return p

    def start_probability(self, start_state):
        return self.start_p[start_state]

    def emission_probability(self, hidden_state, observed_state):
        if hidden_state in self.emission_p and observed_state in self.emission_p[hidden_state]:
            p = self.emission_p[hidden_state][observed_state]
            if p != 0:
                return p
        return .0001

    # adapted pseudocode from wikipedia entry
    def viterbi(self, obs_sequence, hidden_states):
        T1 = {}
        T2 = {}
        # ensure all index keys exist:
        for i in range(len(obs_sequence)):
            T1[i] = {}
            T2[i] = {}

        # unlike pseudocode, use index 0 for first element instead of 1
        for state in hidden_states:
            T1[0][state] = self.start_probability(state) * self.emission_probability(state, obs_sequence[0])
            T2[0][state] = None
        for prev_index, observation in enumerate(obs_sequence[1:]):
            curr_index = prev_index+1
            for state in hidden_states:
                # get the state 'k' that maximizes the value "T1[prev_index][k] * transition_probability(k, state)"
                max_val = None
                k = None
                for potential_k in hidden_states:
                    potential_value = T1[prev_index][potential_k] * self.transition_probability(potential_k, state)
                    if k == None:
                        k = potential_k
                        max_val = potential_value
                    elif potential_value > max_val:
                        k = potential_k
                        max_val = potential_value
                # now run the calculations as in pseudocode
                T1[curr_index][state] = self.emission_probability(state, observation) * max_val # probability maximizing state of subsequence
                T2[curr_index][state] = k # backpointer to previous state

        #backtrack to get best sequence of states:
        sequence = []
        # get best final probability to backtrack from:
        best_tail = None
        i = len(obs_sequence)-1
        for state in hidden_states:
            if best_tail == None:
                best_tail = state
            elif T1[i][best_tail] < T1[i][state]:
                best_tail = state
        # iterate through backpointers:
        current_state = best_tail
        while current_state != None:
            sequence.append(current_state)
            current_state = T2[i][current_state]
            i -= 1
        sequence.reverse()
        return sequence