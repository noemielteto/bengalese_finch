import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    data = {}
    subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']

    for subject in subjects:
        data_label = subject
        data[data_label] = []
        with open(subject + '.txt') as f:
            contents = f.read()
        # we skip first and last separator because they produce empty strings
        for bout in contents.split('Y')[1:-1]:
            # data[data_label].append(['START'] + list(bout) + ['STOP'])
            data[data_label].append(list(bout))

    return data

def flatten(t):
    return [item for sublist in t for item in sublist]

def normalize_matrix(T):
    if not isinstance(T, np.ndarray):
        T = np.array(T)  # TODO will only work for vector!

    # if vector
    if len(T.shape)==1:
        return T/T.sum()
    # if matrix
    else:
        return T / T.sum(axis=1)[:, np.newaxis]

def edit_distance(s1, s2):
    # https://stackoverflow.com/questions/2460177/edit-distance-in-python
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def compute_stationary_distr(bouts, E):

    init_unigram_counts = dict([(e,0) for e in E])
    unigram_counts = dict([(e,0) for e in E])
    bigram_counts = dict([(e_0+e_1,0) for e_0 in E for e_1 in E])
    for bout in bouts:

        if len(bout)==0:
            continue

        init_unigram_counts[bout[0]]+=1

        a = np.unique(list(bout),return_counts=True)
        bout_unigram_counts = dict(zip(a[0], a[1]))
        for unigram, count in bout_unigram_counts.items():
            # ignore very rare syllables
            if unigram in E:
                unigram_counts[unigram] += count

        a = np.unique([''.join(bout[s:s+2]) for s in range(len(bout)-1)],return_counts=True)
        bout_bigram_counts = dict(zip(a[0], a[1]))
        for bigram, count in bout_bigram_counts.items():
            # ignore very rare syllables
            if bigram[0] in E and bigram[1] in E:
                bigram_counts[bigram] += count

    init_unigram_counts = [init_unigram_counts[e] for e in E]
    unigram_counts      = [unigram_counts[e] for e in E]

    bigram_counts_d = np.zeros((len(E),len(E)))
    for bigram, count in bigram_counts.items():
        x,y = E.index(bigram[0]),E.index(bigram[1])
        bigram_counts_d[x][y] = count

    init_unigram_marginals = normalize_matrix(init_unigram_counts)
    unigram_marginals = normalize_matrix(unigram_counts)
    bigram_marginals = normalize_matrix(bigram_counts_d)

    return init_unigram_marginals, unigram_marginals, bigram_marginals
