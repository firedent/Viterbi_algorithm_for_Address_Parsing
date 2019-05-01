import re
from collections import defaultdict, deque
import math
from copy import deepcopy

class TransitionProbability:
    def __init__(self, state_flie_name, q3=False):
        # with open(state_flie_name) as f:
        #     self.num_states = int(f.readline())
        #
        #     # 这部分好像没有用
        #     #
        #     self.states_list = []
        #     self.states_dict = dict()
        #     for i in range(self.num_states):
        #         state_name = f.readline().strip()
        #         self.states_list.append(state_name)
        #         self.states_dict[state_name] = i
        #
        #     self.num_f1_to_f2 = defaultdict(int)
        #     self.num_f1 = defaultdict(int)
        #     for l in f:
        #         f1, f2, f3 = map(int, l.split())
        #         self.num_f1[f1] += 1
        #         self.num_f1_to_f2[(f1, f2)] += 1
        if q3:
            self.knn = 0.0001
        else:
            self.knn = 1
        self.num_states, self.states_list, self.states_dict, self.num_f1, self.num_f1_to_f2 \
            = tool_read_file(state_flie_name)
        self.transition_probability = dict()
        self.BEGIN_INDEX = self.states_dict['BEGIN']
        self.END_INDEX = self.states_dict['END']

    def __getitem__(self, key):
        if key[1] == self.BEGIN_INDEX or key[0] == self.END_INDEX:
            return 0
        else:
            return self.transition_probability.setdefault(key, (self.num_f1_to_f2[key] + self.knn) /
                                                          (self.num_f1[key[0]] + self.knn * self.num_states - 1))

    def get_start_probability(self, state):
        return self[(self.BEGIN_INDEX, state)]

    def get_probability(self, key):
        return self[key]


class EmissionProbabilities:
    def __init__(self, symbol_file_name, end_state_id=-1, q3=False):
        if q3:
            self.knn = 0.0001
        else:
            self.knn = 1
        with open(symbol_file_name) as f:
            self.num_symbols = int(f.readline())

            self.symbols_name_list = []
            self.symbols_dict = dict()
            for i in range(self.num_symbols):
                symbol_name = f.readline().strip()
                self.symbols_name_list.append(symbol_name)
                self.symbols_dict[symbol_name] = i

            self.num_f1_to_f2 = defaultdict(int)
            self.num_f1 = defaultdict(int)
            for l in f:
                f1, f2, f3 = map(int, l.split())
                self.num_f1[f1] += f3
                self.num_f1_to_f2[(f1, f2)] = f3
        # self.num_symbols, self.symbols_name_list, self.symbols_dict, self.num_f1, self.num_f1_to_f2 \
        #     = tool_read_file(symbol_file_name)

        self.emission_probability = {}
        self.END_STATE_ID = end_state_id

    def __getitem__(self, key):
        if key[1] in self.symbols_dict:
            if key[0] != self.END_STATE_ID:
                return self.emission_probability.setdefault(key, (self.num_f1_to_f2[(key[0], self.symbols_dict[key[1]])] + self.knn) /
                                                            (self.num_f1[key[0]] + self.knn * self.num_symbols + 1))
            else:
                return 0
        else:
            if key[0] != self.END_STATE_ID:
                if key[1] != 'ZSC1994END':
                    return self.emission_probability.setdefault(key, self.knn / (self.num_f1[key[0]] + self.knn * self.num_symbols + 1))
                else:
                    return 0
            else:
                if key[1] == 'ZSC1994END':
                    return 1
                else:
                    return 0

    def get_probability(self, key):
        return self[key]


def tool_read_file(filename):
    with open(filename) as f:
        item_num = int(f.readline())

        item_list = []
        item_dict = dict()
        for i in range(item_num):
            state_name = f.readline().strip()
            item_list.append(state_name)
            item_dict[state_name] = i

        num_f1_to_f2 = defaultdict(int)
        num_f1 = defaultdict(int)
        for l in f:
            f1, f2, f3 = map(int, l.split())
            num_f1[f1] += f3
            num_f1_to_f2[(f1, f2)] = f3

        return item_num, item_list, item_dict, num_f1, num_f1_to_f2


def tokens(s):
    tokens = []
    for r in re.split('([,()/\-&])', s):
        tokens += r.split()
    return tokens


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for y in states:
        V[0][y] = start_p(y) * emit_p((y, obs[0]))
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                [(V[t-1][y0] * trans_p((y0, y)) * emit_p((y, obs[t])), y0) for y0 in states]
            )

            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath

    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return path[state], prob


def viterbi_top_k(obs, states, start_p, trans_p, emit_p, k=1):
    V = [{}]
    path = defaultdict()

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = [(start_p(y) * emit_p((y, obs[0])),)]
        path[y] = deque([[y]])

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = defaultdict(deque)

        for y in states:
            tmp_path = deepcopy(path)
            tmp = []
            for y0 in states:
                for b_top in V[t - 1][y0]:
                    tmp.append((b_top[0] * trans_p((y0, y)) * emit_p((y, obs[t])), y0))
            b_top_list = sorted(tmp, key=lambda x: (x[0], -x[1]), reverse=True)[:k]

            V[t][y] = b_top_list
            for b_top in V[t][y]:
                newpath[y].append(tmp_path[b_top[1]].popleft() + [y])

        # Don't need to remember the old paths
        path = newpath

    b_top_list = sorted([(b_top[0], y) for y0 in states for b_top in V[len(obs) - 1][y0]], key=lambda x: (x[0], -x[1]), reverse=True)[:k]
    result = []
    for b_top in b_top_list:
        result.append((path[b_top[1]].popleft(), b_top[0]))
    return result


# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    tp = TransitionProbability(State_File)
    ep = EmissionProbabilities(Symbol_File, end_state_id=tp.END_INDEX)
    result = list()
    with open(Query_File)as f:
        for l in f:
            t = tokens(l.strip())
            t.append('ZSC1994END')
            # print(t)
            path, probability = viterbi(t, range(tp.num_states), tp.get_start_probability, tp.get_probability, ep.get_probability)
            path.append(math.log(probability))
            result.append([tp.BEGIN_INDEX]+path)
    # # 1
    # print(ep[(25, 'ZSC1994END')])
    # # 0
    # print(ep[(423432443, 'ZSC1994END')])
    # # 0
    # print(ep[(9, 'ZSC1994END')])
    # # 0
    # print(ep[(423343343, 'ZSC1994END')])
    # # 0
    # print(ep[(25, 'Kingsford')])
    # # 0
    # print(ep[(25, 'dsadas')])
    # # n/
    # print(ep[(9, 'Kingsford')])
    # # 1/
    # print(ep[(9, 'sadsadas')])
    return result


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):  # do not change the heading of the function
    tp = TransitionProbability(State_File)
    ep = EmissionProbabilities(Symbol_File, end_state_id=tp.END_INDEX)
    result = list()
    if k == 1:
        with open(Query_File)as f:
            for l in f:
                t = tokens(l.strip())
                t.append('ZSC1994END')
                # print(t)
                path, probability = viterbi(t, range(tp.num_states), tp.get_start_probability, tp.get_probability,
                                            ep.get_probability)
                path.append(math.log(probability))
                result.append([tp.BEGIN_INDEX] + path)
        return result
    else:
        with open(Query_File)as f:
            for l in f:
                t = tokens(l.strip())
                t.append('ZSC1994END')
                # print(t)
                b_top_list = viterbi_top_k(t, range(tp.num_states), tp.get_start_probability, tp.get_probability, ep.get_probability, k)
                for b in b_top_list:
                    sub_list = [tp.BEGIN_INDEX]
                    sub_list += b[0]
                    sub_list.append(math.log(b[1]))
                    result.append(sub_list)
        return result

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    tp = TransitionProbability(State_File)
    ep = EmissionProbabilities(Symbol_File, end_state_id=tp.END_INDEX)
    result = list()
    with open(Query_File)as f:
        for l in f:
            t = tokens(l.strip())
            t.append('ZSC1994END')
            # print(t)
            path, probability = viterbi(t, range(tp.num_states), tp.get_start_probability, tp.get_probability, ep.get_probability)
            # path[-2] = 6
            path.append(math.log(probability))
            result.append([tp.BEGIN_INDEX] + path)
    return result


# prefix = './toy_example/'
# prefix1 = './dev_set/'
# files = ['State_File', 'Symbol_File', 'Query_File']
#
# t1 = time.time()
# viterbi_result = advanced_decoding(*list(map(lambda x: prefix1+x, files)))
# t2 = time.time()
# print(t2-t1)
#
#
# with open('./p.txt', mode='w+') as p:
#     for row in viterbi_result:
        # print(row)
        # p.write(' '.join(map(str, row[:-1]))+'\n')
        # p.write(' '.join(map(str, row[:-1]))+'\n')

